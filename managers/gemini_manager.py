import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from managers.chroma_manager import ChromaManager

load_dotenv()
logger = logging.getLogger(__name__)

class GeminiManager:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        
        # Define system instruction with context
        system_instruction = """You are a professional couple counselor AI assistant. 
        Use the following context to provide helpful guidance:
        
        {context}
        
        Always maintain a supportive, non-judgmental tone and provide evidence-based advice."""
        
        self.model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=system_instruction)
        self.chat_sessions = {}
        
        # Initialize ChromaDB for RAG
        self.chroma_manager = ChromaManager()

    def start_chat(self, client_id: int):
        # Get previous messages from database
        from managers.db_manager import DBManager
        messages = DBManager.get_messages(client_id)
        
        # Convert to Gemini history format
        history = []
        for msg in messages:
            role = "user" if msg["sender"] == "user" else "model"
            history.append({"role": role, "parts": [msg["message"]]})
        
        self.chat_sessions[client_id] = self.model.start_chat(history=history)
        logger.info(f"Started Gemini session for client {client_id} with {len(history)} messages")

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
            
            response = self.chat_sessions[client_id].send_message(enhanced_message, stream=False)
            return response.text
        except Exception as e:
            logger.error(f"Gemini error: {str(e)}")
            return "Sorry, I encountered an error processing your request."

    def remove_chat(self, client_id: int):
        if client_id in self.chat_sessions:
            del self.chat_sessions[client_id]
            logger.info(f"Removed Gemini session for client {client_id}")
