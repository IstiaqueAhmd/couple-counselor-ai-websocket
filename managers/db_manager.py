import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime

load_dotenv()
logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI)
db = client["chat_db"]  # or use your actual DB name
collection = db["chats"]

users = db["users"]
class DBManager:
    @staticmethod
    def save_message(client_id: int, sender: str, message: str):
        doc = {
            "client_id": client_id,
            "sender": sender,
            "message": message,
            "timestamp": datetime.utcnow()
        }
        collection.insert_one(doc)
    
    @staticmethod
    def save_user(user_id: int, name: str, age: int, gender: str, spouse_id: int = None):
        doc = {
            "user_id": user_id,
            "gender": gender,
            "spouse_id": spouse_id,
            "name": name,
            "age": age,
        }
        users.insert_one(doc)
        
    @staticmethod
    def get_messages(client_id: int):
        return list(collection.find({"client_id": client_id}))
    @staticmethod
    def get_user(user_id: int):
        """Get user information by user_id"""
        return users.find_one({"user_id": user_id})

    @staticmethod
    def get_spouse(user_id: int):
        """Get spouse information with bidirectional lookup"""
        user = users.find_one({"user_id": user_id})
        if user and user.get("spouse_id"):
            # Direct spouse_id lookup
            return users.find_one({"user_id": user["spouse_id"]})
        else:
            # Reverse lookup - check if user_id is someone's spouse_id
            spouse = users.find_one({"spouse_id": user_id})
            return spouse
        return None

    @staticmethod
    def link_spouses(user_id_1: int, user_id_2: int):
        """Link two users as spouses (bidirectional)"""
        try:
            users.update_one(
                {"user_id": user_id_1},
                {"$set": {"spouse_id": user_id_2}}
            )
            users.update_one(
                {"user_id": user_id_2},
                {"$set": {"spouse_id": user_id_1}}
            )
            return True
        except Exception as e:
            logger.error(f"Error linking spouses {user_id_1} and {user_id_2}: {str(e)}")
            return False