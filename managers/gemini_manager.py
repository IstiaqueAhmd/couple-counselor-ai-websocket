import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class GeminiManager:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.chat_sessions = {}

    def start_chat(self, client_id: int):
        self.chat_sessions[client_id] = self.model.start_chat(history=[])
        logger.info(f"Started new Gemini session for client {client_id}")

    def get_response(self, client_id: int, message: str) -> str:
        if client_id not in self.chat_sessions:
            self.start_chat(client_id)
        try:
            response = self.chat_sessions[client_id].send_message(message, stream=False)
            return response.text
        except Exception as e:
            logger.error(f"Gemini error: {str(e)}")
            return "Sorry, I encountered an error processing your request."

    def remove_chat(self, client_id: int):
        if client_id in self.chat_sessions:
            del self.chat_sessions[client_id]
            logger.info(f"Removed Gemini session for client {client_id}")
