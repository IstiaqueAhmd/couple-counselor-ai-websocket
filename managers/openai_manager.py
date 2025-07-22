import os
import logging
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from managers.chroma_manager import ChromaManager
from typing import List, Dict
from managers.db_manager import DBManager

load_dotenv()
logger = logging.getLogger(__name__)

class OpenAIManager:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key, timeout=30.0, max_retries=3)
        self.chat_sessions = {}
        self.max_tokens = 100000
        self.recent_messages_count = 10
        
        # Initialize token encoder with fallback
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        except Exception:
            self.encoding = None
        
        self.chroma_manager = ChromaManager()
        logger.info("OpenAI manager initialized successfully")

    def count_tokens(self, messages: List[Dict]) -> int:
        """Count tokens in message list"""
        if not self.encoding:
            total_chars = sum(len(msg.get("content", "")) for msg in messages)
            return total_chars // 4
        
        total = 0
        for message in messages:
            total += len(self.encoding.encode(message.get("content", "")))
            total += 4  # Role overhead
        return total

    def trim_conversation_history(self, messages: List[Dict]) -> List[Dict]:
        """Trim conversation to fit within token limits"""
        if not messages or len(messages) <= self.recent_messages_count:
            return messages
        
        # Keep system message and recent messages
        system_msg = messages[0] if messages[0]["role"] == "system" else None
        recent_msgs = messages[-self.recent_messages_count:]
        older_msgs = messages[1:-self.recent_messages_count] if system_msg else messages[:-self.recent_messages_count]
        
        # Start with system and recent messages
        result = ([system_msg] if system_msg else []) + recent_msgs
        current_tokens = self.count_tokens(result)
        
        # Add older messages if they fit
        for msg in reversed(older_msgs):
            msg_tokens = self.count_tokens([msg])
            if current_tokens + msg_tokens > self.max_tokens:
                break
            result.insert(-len(recent_msgs), msg)
            current_tokens += msg_tokens
        
        logger.info(f"Trimmed conversation: {len(messages)} -> {len(result)} messages")
        return result

    def start_chat(self, client_id: int):
        """Initialize chat session with conversation history"""
        messages = DBManager.get_messages(client_id)
        user = DBManager.get_user(client_id)
        
        # Create conversation history
        history = [{
            "role": "system",
            "content": f"""You are a professional couple counselor AI assistant talking with {user['name'] if user else 'User'}. 
            Provide personalized, evidence-based guidance with a supportive, non-judgmental tone."""
        }]
        
        for msg in messages:
            role = "user" if msg["sender"] == "user" else "assistant"
            history.append({"role": role, "content": msg["message"]})
        
        self.chat_sessions[client_id] = history
        
        # Only store in ChromaDB if this is new conversation data that hasn't been stored yet
        # Remove this automatic storage since messages should already be in ChromaDB
        # if len(history) > 1:
        #     self.chroma_manager.store_conversation_chunk(client_id, history[1:])
        
        logger.info(f"Started chat for client {client_id} with {len(history)} messages")

    def get_context(self, client_id: int, message: str) -> Dict[str, str]:
        """Get relevant context from knowledge base and conversation history"""
        try:
            search_results = self.chroma_manager.hybrid_search(
                client_id=client_id,
                query=message,
                knowledge_results=3,
                conversation_results=3
            )
            
            context = {
                "long_term_memory": "",
                "partner_history": "",
                "relevant_knowledge": "",
                "current_message": message
            }
            
            # Long-term memory from conversation history
            if search_results["conversation_history"]:
                # Remove duplicates and format properly
                seen_content = set()
                memory_items = []
                for result in search_results["conversation_history"]:
                    content = result['content'].strip()
                    if content not in seen_content and len(content) > 0:
                        seen_content.add(content)
                        memory_items.append(f"- {content}")
                
                # Limit to most relevant items
                context["long_term_memory"] = "\n".join(memory_items[:5])
            
            # Partner conversation history
            if search_results["partner_history"]:
                seen_content = set()
                partner_items = []
                for result in search_results["partner_history"]:
                    content = result['content'].strip()
                    if content not in seen_content and len(content) > 0:
                        seen_content.add(content)
                        partner_items.append(f"- {content}")
                
                context["partner_history"] = "\n".join(partner_items[:5])
            
            # Relevant knowledge base information
            if search_results["knowledge_base"]:
                knowledge_items = []
                for result in search_results["knowledge_base"]:
                    knowledge_items.append(f"- {result['content']}")
                context["relevant_knowledge"] = "\n".join(knowledge_items)
            
            return context
            
        except Exception as e:
            logger.error(f"Context retrieval error: {str(e)}")
            return {
                "long_term_memory": "",
                "partner_history": "",
                "relevant_knowledge": "",
                "current_message": message
            }

    def create_context_prompt(self, context: Dict[str, str]) -> str:
        """Create a structured context prompt for the AI"""
        prompt_parts = []
        
        if context["long_term_memory"]:
            prompt_parts.append(f"""LONG-TERM MEMORY (Previous conversations with this user):
{context["long_term_memory"]}""")
        
        if context["relevant_knowledge"]:
            prompt_parts.append(f"""RELEVANT KNOWLEDGE (Professional guidance):
{context["relevant_knowledge"]}""")
        
        prompt_parts.append(f"""CURRENT MESSAGE: {context["current_message"]}
                            
        Please respond to the current message while being aware of:
        1. SHORT-TERM MEMORY: Recent conversation context (last few messages)
        2. LONG-TERM MEMORY: Important details from past sessions shown above
        3. RELEVANT KNOWLEDGE: Professional guidance that applies to this situation

        Provide a personalized response that acknowledges continuity while addressing the current concern.""")
        
        return "\n\n".join(prompt_parts)

    def get_response(self, client_id: int, message: str) -> str:
        """Get AI response for user message"""
        if client_id not in self.chat_sessions:
            self.start_chat(client_id)
        
        try:
            # Get context information
            context = self.get_context(client_id, message)
            
            # Add user message to history first
            self.chat_sessions[client_id].append({
                "role": "user",
                "content": message
            })
            
            # Create enhanced system message with context
            base_system_msg = self.chat_sessions[client_id][0]["content"]
            enhanced_system_msg = self.create_enhanced_system_message(base_system_msg, context)
            
            # Prepare messages for API call
            current_messages = self.chat_sessions[client_id].copy()
            current_messages[0]["content"] = enhanced_system_msg  # Replace system message
            
            # Remove duplicate consecutive messages before trimming
            cleaned_messages = self.remove_duplicate_messages(current_messages)
            
            # Trim and get response
            trimmed_messages = self.trim_conversation_history(cleaned_messages)
            print(trimmed_messages)

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=trimmed_messages,
                max_tokens=1500,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # Add AI response to history
            self.chat_sessions[client_id].append({
                "role": "assistant",
                "content": ai_response
            })
            
            # Store the exchange in ChromaDB
            new_messages = [
                {"role": "user", "content": message},
                {"role": "assistant", "content": ai_response}
            ]
            self.chroma_manager.store_conversation_chunk(client_id, new_messages)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"OpenAI error for client {client_id}: {str(e)}")
            return "Sorry, I encountered an error processing your request."

    def create_enhanced_system_message(self, base_message: str, context: Dict[str, str]) -> str:
        """Create a single enhanced system message with context"""
        enhanced_parts = [base_message]
        
        if context["long_term_memory"]:
            enhanced_parts.append(f"\nRELEVANT CONVERSATION HISTORY (This User):\n{context['long_term_memory']}")
        
        if context["partner_history"]:
            enhanced_parts.append(f"\nRELEVANT PARTNER CONVERSATION HISTORY (Their Partner):\n{context['partner_history']}")
        
        if context["relevant_knowledge"]:
            enhanced_parts.append(f"\nRELEVANT PROFESSIONAL KNOWLEDGE:\n{context['relevant_knowledge']}")
        
        enhanced_parts.append("\nUse this context to provide personalized, continuous care while addressing the current message. Consider both individual and relationship dynamics when responding.")
        
        return "\n".join(enhanced_parts)

    def remove_duplicate_messages(self, messages: List[Dict]) -> List[Dict]:
        """Remove consecutive duplicate messages while preserving system message"""
        if len(messages) <= 1:
            return messages
        
        cleaned = [messages[0]]  # Always keep system message
        
        for i in range(1, len(messages)):
            current = messages[i]
            previous = cleaned[-1]
            
            # Don't add if it's identical to the previous message
            if not (current["role"] == previous["role"] and 
                   current["content"] == previous["content"]):
                cleaned.append(current)
        
        return cleaned
    def remove_chat(self, client_id: int):
        if client_id in self.chat_sessions:
            del self.chat_sessions[client_id]
            logger.info(f"Removed OpenAI session for client {client_id}")