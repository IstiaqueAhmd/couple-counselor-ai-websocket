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
            
            # Get recent messages to exclude from search results
            recent_messages_content = set()
            if client_id in self.chat_sessions:
                recent_messages = self.chat_sessions[client_id][-self.recent_messages_count:]
                for msg in recent_messages:
                    recent_messages_content.add(msg.get("content", "").strip())
            
            # Long-term memory from conversation history
            if search_results["conversation_history"]:
                # Remove duplicates and recent messages
                seen_content = set()
                memory_items = []
                for result in search_results["conversation_history"]:
                    content = result['content'].strip()
                    if (content not in seen_content and 
                        content not in recent_messages_content and 
                        len(content) > 0):
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
            
            # Prepare messages for API call with separate context system messages
            current_messages = self.chat_sessions[client_id].copy()
            
            # Insert context system messages after the base system message
            context_messages = self.create_context_system_messages(context)
            
            # Insert context messages right after the base system message (index 1)
            for i, context_msg in enumerate(context_messages):
                current_messages.insert(1 + i, context_msg)

            # Trim and get response
            trimmed_messages = self.trim_conversation_history(current_messages)
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

    def create_context_system_messages(self, context: Dict[str, str]) -> List[Dict[str, str]]:
        """Create separate system messages for each context type"""
        context_messages = []
        
        if context["long_term_memory"]:
            context_messages.append({
                "role": "system",
                "content": f"YOUR PREVIOUS CONVERSATION WITH THIS USER:\n{context['long_term_memory']}"
            })
        
        if context["partner_history"]:
            context_messages.append({
                "role": "system", 
                "content": f"YOUR PREVIOUS CONVERSATION WITH THIS USER'S PARTNER:\n{context['partner_history']}"
            })
        
        if context["relevant_knowledge"]:
            context_messages.append({
                "role": "system",
                "content": f"RELEVANT PROFESSIONAL KNOWLEDGE:\n{context['relevant_knowledge']}"
            })
        
        # Add instruction message if any context exists
        if context_messages:
            context_messages.append({
                "role": "system",
                "content": "Use this context to provide personalized, continuous care while addressing the current message. Consider both individual and relationship dynamics when responding."
            })
        
        return context_messages

    def remove_chat(self, client_id: int):
        if client_id in self.chat_sessions:
            del self.chat_sessions[client_id]
            logger.info(f"Removed OpenAI session for client {client_id}")
