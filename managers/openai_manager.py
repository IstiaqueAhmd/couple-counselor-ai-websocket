import os
import logging
from openai import OpenAI
from dotenv import load_dotenv
from managers.chroma_manager import ChromaManager

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
        
        # Initialize ChromaDB for RAG
        try:
            self.chroma_manager = ChromaManager()
            logger.info("ChromaDB manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB manager: {str(e)}")
            raise

    def start_chat(self, client_id: int):
        # Get previous messages from database
        from managers.db_manager import DBManager
        messages = DBManager.get_messages(client_id)
        
        # Convert to OpenAI history format
        history = [
            {
                "role": "system",
                "content": """You are a professional couple counselor AI assistant. 
                Use the following context to provide helpful guidance when available.
                Always maintain a supportive, non-judgmental tone and provide evidence-based advice."""
            }
        ]
        
        for msg in messages:
            role = "user" if msg["sender"] == "user" else "assistant"
            history.append({"role": role, "content": msg["message"]})
        
        self.chat_sessions[client_id] = history
        logger.info(f"Started OpenAI session for client {client_id} with {len(history)} messages")

    def get_rag_context(self, message: str) -> str:
        """Get relevant context using ChromaDB RAG"""
        try:
            # Search for relevant documents
            search_results = self.chroma_manager.search(message, n_results=3)
            if not search_results:
                return ""
            
            # Format context from search results
            context_parts = []
            for i, result in enumerate(search_results, 1):
                context_parts.append(f"Source {i}: {result['content']}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"RAG context retrieval error: {str(e)}")
            return ""

    def get_response(self, client_id: int, message: str) -> str:
        if client_id not in self.chat_sessions:
            self.start_chat(client_id)
        
        try:
            # Get relevant context using RAG
            rag_context = self.get_rag_context(message)
            
            # Add context to the message if available
            if rag_context:
                enhanced_message = f"Context: {rag_context}\n\nUser Query: {message}"
            else:
                enhanced_message = message
            
            # Add the user message to the conversation history
            self.chat_sessions[client_id].append({
                "role": "user",
                "content": enhanced_message
            })
            
            # Get response from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # You can change this to gpt-4 or other models
                messages=self.chat_sessions[client_id],
                max_tokens=1500,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # Add the AI response to the conversation history
            self.chat_sessions[client_id].append({
                "role": "assistant",
                "content": ai_response
            })
            
            return ai_response
            
        except Exception as e:
            logger.error(f"OpenAI error: {str(e)}")
            return "Sorry, I encountered an error processing your request."

    def remove_chat(self, client_id: int):
        if client_id in self.chat_sessions:
            del self.chat_sessions[client_id]
            logger.info(f"Removed OpenAI session for client {client_id}")
