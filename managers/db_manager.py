import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI)
db = client["chat_db"]  # or use your actual DB name
collection = db["chats"]
couples = db["couples"]
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
    def save_couple(couple_id: str, husband_id: int, wife_id: int):
        doc = {
            "couple_id": couple_id,
            "husband_id": husband_id,
            "wife_id": wife_id,
            "timestamp": datetime.utcnow()
        }
        couples.insert_one(doc)
    @staticmethod
    def save_user(user_id: int, name: str, age: int, gender: str):
        doc = {
            "user_id": user_id,
            "name": name,
            "age": age,
        }
        users.insert_one(doc)
        
    @staticmethod
    def get_messages(client_id: int):
        return list(collection.find({"client_id": client_id}))
    @staticmethod
    def get_couple(couple_id: str):
        return couples.find_one({"couple_id": couple_id})