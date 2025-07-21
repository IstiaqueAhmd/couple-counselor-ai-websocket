import logging
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from managers.db_manager import DBManager
from managers.gemini_manager import GeminiManager
from managers.connection_manager import ConnectionManager
from managers.chroma_manager import ChromaManager
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
chroma_manager = ChromaManager()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    """Ingest PDF or TXT files into the knowledge base"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.txt')):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF and TXT files are supported"
            )
        
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Ingest into ChromaDB
        metadata = {
            "upload_date": str(datetime.now()),
            "file_size": len(content)
        }
        
        success = chroma_manager.ingest_document(file_path, metadata)
        
        if success:
            logger.info(f"Successfully ingested file: {file.filename}")
            return {
                "message": f"File '{file.filename}' successfully ingested into knowledge base",
                "filename": file.filename,
                "file_size": len(content)
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to ingest file into knowledge base"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/knowledge-base/stats")
async def get_knowledge_base_stats():
    """Get statistics about the knowledge base"""
    try:
        stats = chroma_manager.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting knowledge base stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/knowledge-base/{filename}")
async def delete_file_from_knowledge_base(filename: str):
    """Delete a specific file from the knowledge base"""
    try:
        success = chroma_manager.delete_documents_by_filename(filename)
        if success:
            return {"message": f"File '{filename}' deleted from knowledge base"}
        else:
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
    from datetime import datetime
    uvicorn.run(app, host="127.0.0.1", port=3000)
