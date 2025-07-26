import logging
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from managers.db_manager import DBManager
from managers.openai_manager import OpenAIManager
from managers.connection_manager import ConnectionManager
from managers.chroma_manager import ChromaManager
from utils.helpers import create_json_message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request bodies
class MessageData(BaseModel):
    message: str
    sender: str = "admin"  # Default sender is 'admin'

class UserData(BaseModel):
    user_id: int
    name: str
    age: int
    gender: str
    spouse_id: int = None

# Initialize FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Managers
openai_manager = OpenAIManager()
connection_manager = ConnectionManager()
chroma_manager = ChromaManager()

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

            ai_response = openai_manager.get_response(client_id, message)

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
        openai_manager.remove_chat(client_id)

    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        connection_manager.disconnect(client_id)
        openai_manager.remove_chat(client_id)

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

@app.post("/send-message/{client_id}")
async def send_personal_message(client_id: int, message_data: MessageData):
    """Send a personal message to a specific client"""
    try:
        # Check if client is connected
        if client_id not in connection_manager.active_connections:
            raise HTTPException(
                status_code=404, 
                detail=f"Client {client_id} is not connected"
            )
        
        # Send the message
        await connection_manager.send_personal_message(
            create_json_message(message_data.sender, message_data.message),
            client_id
        )
        
        # Save the message to database
        DBManager.save_message(client_id, message_data.sender, message_data.message)
        
        logger.info(f"Sent personal message to client {client_id} from {message_data.sender}")
        
        return {
            "message": f"Message sent successfully to client {client_id}",
            "client_id": client_id,
            "sender": message_data.sender,
            "content": message_data.message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending personal message to client {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")

@app.get("/clients")
async def get_active_clients():
    """Get list of active connected clients"""
    try:
        active_clients = list(connection_manager.active_connections.keys())
        return {
            "active_clients": active_clients,
            "count": len(active_clients)
        }
    except Exception as e:
        logger.error(f"Error getting active clients: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    


"################################### Document ingestion Start ###########################################"

@app.post("/documents/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Ingest a document (PDF or TXT) into the knowledge base"""
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ['.pdf', '.txt']:
            raise HTTPException(
                status_code=400, 
                detail="Only PDF and TXT files are supported"
            )
        
        # Create uploads directory if it doesn't exist
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save uploaded file temporarily
        file_path = os.path.join(uploads_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Ingest document using ChromaManager
        success = chroma_manager.ingest_document(file_path)
        
        if success:
            logger.info(f"Successfully ingested document: {file.filename}")
            
            # Get collection stats
            stats = chroma_manager.get_collection_stats()
            
            return {
                "message": f"Document '{file.filename}' ingested successfully",
                "filename": file.filename,
                "file_type": file_extension,
                "file_size": len(content),
                "collection_stats": stats
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to ingest document: {file.filename}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/documents/stats")
async def get_knowledge_base_stats():
    """Get knowledge base statistics"""
    try:
        stats = chroma_manager.get_collection_stats()
        return {
            "knowledge_base": stats,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting knowledge base stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/search")
async def search_knowledge_base(query: str, n_results: int = 3):
    """Search the knowledge base"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        results = chroma_manager.search_knowledge_base(query, n_results)
        
        return {
            "query": query,
            "results": results,
            "total_results": len(results)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


"################################### Document ingestion End ###########################################"


@app.post("/users/save")
async def save_user(user_data: UserData):
    """Save user information"""
    try:
        DBManager.save_user(
            user_data.user_id, 
            user_data.name, 
            user_data.age, 
            user_data.gender, 
            user_data.spouse_id
        )
        return {"message": "User saved successfully"}
    except Exception as e:
        logger.error(f"Error saving user: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}/spouse")
async def get_spouse(user_id: int):
    """Get spouse information for a user"""
    try:
        spouse = DBManager.get_spouse(user_id)
        if spouse:
            return {
                "spouse_id": spouse["user_id"],
                "name": spouse["name"],
                "age": spouse["age"],
                "gender": spouse["gender"]
            }
        return {"message": "Spouse not found"}
    except Exception as e:
        logger.error(f"Error getting spouse information: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)
