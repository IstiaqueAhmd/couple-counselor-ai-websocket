import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from managers.db_manager import DBManager
from managers.gemini_manager import GeminiManager
from managers.connection_manager import ConnectionManager
from utils.helpers import create_json_message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Managers
gemini_manager = GeminiManager()
connection_manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await connection_manager.connect(websocket, client_id)
    try:
        while True:
            message = await websocket.receive_text()
            logger.info(f"Received message from client {client_id}: {message}")
            
            # Save user message
            DBManager.save_message(client_id, "user", message)

            ai_response = gemini_manager.get_response(client_id, message)

            # Save AI response
            DBManager.save_message(client_id, "ai", ai_response)

            await connection_manager.send_personal_message(
                create_json_message("ai", ai_response),
                client_id
            )

            logger.info(f"Sent response to client {client_id}")

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
        connection_manager.disconnect(client_id)
        gemini_manager.remove_chat(client_id)

    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        connection_manager.disconnect(client_id)
        gemini_manager.remove_chat(client_id)

@app.get("/history/{client_id}")
async def get_history(client_id: int):
    messages = DBManager.get_messages(client_id)
    return [
        {
            "sender": msg["sender"],
            "message": msg["message"],
            "timestamp": msg["timestamp"]
        }
        for msg in messages
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)
