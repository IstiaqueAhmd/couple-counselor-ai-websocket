import json
from datetime import datetime

def create_json_message(sender: str, message: str) -> str:
    return json.dumps({
        "sender": sender,
        "message": message,
        "timestamp": datetime.now().isoformat()
    })
