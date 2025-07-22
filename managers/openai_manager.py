import os
import logging
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from managers.chroma_manager import ChromaManager
from typing import List, Dict

load_dotenv()
logger = logging.getLogger(__name__)

class OpenAIManager:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        try:
            self.client = OpenAI(
                api_key=api_key,
                timeout=30.0,
                max_retries=3
            )
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
        
        self.chat_sessions = {}
        self.max_tokens = 100000  # Leave room for response
        self.recent_messages_count = 10  # Number of recent messages to always include
        
        # Initialize token encoder
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        except Exception as e:
            logger.warning(f"Failed to initialize tiktoken encoder: {str(e)}")
            self.encoding = None
        
        # Initialize ChromaDB for RAG
        try:
            self.chroma_manager = ChromaManager()
            logger.info("ChromaDB manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB manager: {str(e)}")
            raise

    def count_tokens(self, messages: List[Dict]) -> int:
        """Count tokens in message list"""
        if not self.encoding:
            # Fallback estimation: ~4 chars per token
            total_chars = sum(len(msg.get("content", "")) for msg in messages)
            return total_chars // 4
        
        total = 0
        for message in messages:
            # Add tokens for role and content
            total += len(self.encoding.encode(message.get("content", "")))
            total += 4  # Role overhead
        return total

    def trim_conversation_history(self, messages: List[Dict], keep_recent: int = None) -> List[Dict]:
        """Trim conversation to fit within token limits while keeping recent messages"""
        if not messages:
            return messages
        
        keep_recent = keep_recent or self.recent_messages_count
        
        # Always keep system message
        system_msg = messages[0] if messages and messages[0]["role"] == "system" else None
        conversation_msgs = messages[1:] if system_msg else messages
        
        # Always keep the most recent N messages
        recent_msgs = conversation_msgs[-keep_recent:] if len(conversation_msgs) > keep_recent else conversation_msgs
        older_msgs = conversation_msgs[:-keep_recent] if len(conversation_msgs) > keep_recent else []
        
        # Start with system message and recent messages
        result = [system_msg] if system_msg else []
        current_tokens = self.count_tokens(result + recent_msgs)
        
        # Add older messages if they fit
        trimmed_older = []
        for msg in reversed(older_msgs):
            msg_tokens = self.count_tokens([msg])
            if current_tokens + msg_tokens > self.max_tokens:
                break
            trimmed_older.insert(0, msg)
            current_tokens += msg_tokens
        
        # Combine all parts
        result.extend(trimmed_older)
        result.extend(recent_msgs)
        
        logger.info(f"Trimmed conversation: {len(messages)} -> {len(result)} messages ({current_tokens} tokens)")
        return result

    def start_chat(self, client_id: int):
        """Initialize chat session with conversation history including spouse context"""
        # Get previous messages from database
        from managers.db_manager import DBManager
        messages = DBManager.get_messages(client_id)
        
        # Get spouse information for enhanced system prompt
        spouse = DBManager.get_spouse(client_id)
        spouse_info = ""
        if spouse:
            spouse_info = f"\nYou are counseling a couple. The current client's partner is {spouse.get('name', 'their partner')} ({spouse.get('gender', 'unknown')} age {spouse.get('age', 'unknown')}). Consider both partners' perspectives when providing guidance."
        
        # Convert to OpenAI history format
        history = [
            {
                "role": "system",
                "content": f"""You are a professional couple counselor AI assistant. 
                You have access to relevant context from both professional knowledge and previous conversations from both partners.
                Use this context to provide personalized, evidence-based guidance.
                Always maintain a supportive, non-judgmental tone and build upon previous discussions when relevant.{spouse_info}
                
                When you have context from both partners, consider:
                - Different perspectives on the same issues
                - Communication patterns between partners
                - How to help bridge understanding between them
                - Maintaining confidentiality while using insights to help both individuals"""
            }
        ]
        
        for msg in messages:
            role = "user" if msg["sender"] == "user" else "assistant"
            history.append({"role": role, "content": msg["message"]})
        
        self.chat_sessions[client_id] = history
        
        # Store conversation chunks in ChromaDB for future retrieval
        if len(history) > 1:  # More than just system message
            self._store_conversation_chunk(client_id, history[1:])  # Exclude system message
        
        logger.info(f"Started OpenAI session for client {client_id} with {len(history)} messages{' (with spouse context)' if spouse else ''}")

    def _store_conversation_chunk(self, client_id: int, messages: List[Dict]):
        """Store conversation chunk in ChromaDB"""
        try:
            # Store in chunks of 5 messages to maintain context
            chunk_size = 5
            for i in range(0, len(messages), chunk_size):
                chunk = messages[i:i + chunk_size]
                self.chroma_manager.store_conversation_chunk(client_id, chunk)
        except Exception as e:
            logger.error(f"Failed to store conversation chunk for client {client_id}: {str(e)}")

    def get_hybrid_context(self, client_id: int, message: str) -> Dict:
        """Get hybrid context from both knowledge base and conversation history (including spouse)"""
        try:
            # Perform hybrid search for current client
            search_results = self.chroma_manager.hybrid_search(
                client_id=client_id,
                query=message,
                knowledge_results=2,
                conversation_results=2
            )
            
            # Get spouse context if spouse exists
            spouse_context = self._get_spouse_context(client_id, message)
            
            # Format contexts
            context = {
                "knowledge_context": "",
                "conversation_context": "",
                "spouse_context": "",
                "total_sources": search_results["total_results"] + (1 if spouse_context else 0)
            }
            
            # Format knowledge base context
            if search_results["knowledge_base"]:
                knowledge_parts = []
                for i, result in enumerate(search_results["knowledge_base"], 1):
                    knowledge_parts.append(f"Professional Resource {i}: {result['content']}")
                context["knowledge_context"] = "\n\n".join(knowledge_parts)
            
            # Format conversation history context
            if search_results["conversation_history"]:
                history_parts = []
                for i, result in enumerate(search_results["conversation_history"], 1):
                    history_parts.append(f"Previous Discussion {i}: {result['content']}")
                context["conversation_context"] = "\n\n".join(history_parts)
            
            # Add spouse context
            if spouse_context:
                context["spouse_context"] = spouse_context
            
            return context
            
        except Exception as e:
            logger.error(f"Hybrid context retrieval error for client {client_id}: {str(e)}")
            return {"knowledge_context": "", "conversation_context": "", "spouse_context": "", "total_sources": 0}

    def _get_spouse_context(self, client_id: int, message: str) -> str:
        """Get relevant context from spouse's conversations"""
        try:
            from managers.db_manager import DBManager
            
            # Get spouse information
            spouse = DBManager.get_spouse(client_id)
            if not spouse:
                return ""
            
            spouse_id = spouse["user_id"]
            
            # Search spouse's conversation history for relevant messages
            spouse_results = self.chroma_manager.search_conversation_history(
                client_id=spouse_id,
                query=message,
                n_results=2
            )
            
            if spouse_results:
                spouse_parts = []
                spouse_name = spouse.get("name", f"Partner")
                
                for i, result in enumerate(spouse_results, 1):
                    spouse_parts.append(f"{spouse_name}'s Previous Discussion {i}: {result['content']}")
                
                return "\n\n".join(spouse_parts)
            
            return ""
            
        except Exception as e:
            logger.error(f"Error getting spouse context for client {client_id}: {str(e)}")
            return ""

    def format_enhanced_message(self, message: str, context: Dict) -> str:
        """Format message with hybrid context including spouse information"""
        enhanced_parts = []
        
        # Add contexts if available
        if context["knowledge_context"]:
            enhanced_parts.append(f"PROFESSIONAL GUIDANCE:\n{context['knowledge_context']}")
        
        if context["conversation_context"]:
            enhanced_parts.append(f"YOUR PREVIOUS DISCUSSIONS:\n{context['conversation_context']}")
        
        if context["spouse_context"]:
            enhanced_parts.append(f"PARTNER'S RELEVANT DISCUSSIONS:\n{context['spouse_context']}")
        
        # Add the current user query
        enhanced_parts.append(f"CURRENT QUESTION:\n{message}")
        
        if enhanced_parts[:-1]:  # If we have context
            enhanced_parts.insert(-1, "---")  # Separator before current question
        
        return "\n\n".join(enhanced_parts)

    def get_response(self, client_id: int, message: str) -> str:
        if client_id not in self.chat_sessions:
            self.start_chat(client_id)
        
        try:
            # Get hybrid context (knowledge base + conversation history)
            context = self.get_hybrid_context(client_id, message)
            
            # Format enhanced message with context
            if context["total_sources"] > 0:
                enhanced_message = self.format_enhanced_message(message, context)
                logger.info(f"Enhanced message for client {client_id} with {context['total_sources']} context sources")
            else:
                enhanced_message = message
                logger.info(f"No additional context found for client {client_id}")
            
            # Add the user message to the conversation history
            self.chat_sessions[client_id].append({
                "role": "user",
                "content": enhanced_message
            })
            
            # Trim conversation history if needed
            trimmed_messages = self.trim_conversation_history(self.chat_sessions[client_id])
            
            # Get response from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=trimmed_messages,
                max_tokens=1500,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # Add the AI response to the conversation history
            self.chat_sessions[client_id].append({
                "role": "assistant",
                "content": ai_response
            })
            
            # Store the new exchange in ChromaDB
            new_messages = [
                {"role": "user", "content": message},  # Store original message, not enhanced
                {"role": "assistant", "content": ai_response}
            ]
            self._store_conversation_chunk(client_id, new_messages)
            
            return ai_response
            
        except Exception as e:
            logger.error(f"OpenAI error for client {client_id}: {str(e)}")
            return "Sorry, I encountered an error processing your request."

    def remove_chat(self, client_id: int):
        if client_id in self.chat_sessions:
            del self.chat_sessions[client_id]
            logger.info(f"Removed OpenAI session for client {client_id}")

    # Legacy method for backward compatibility
    def get_rag_context(self, message: str) -> str:
        """Legacy RAG context method - now uses knowledge base only"""
        try:
            search_results = self.chroma_manager.search_knowledge_base(message, n_results=3)
            if not search_results:
                return ""
            
            context_parts = []
            for i, result in enumerate(search_results, 1):
                context_parts.append(f"Source {i}: {result['content']}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"RAG context retrieval error: {str(e)}")
            return ""
