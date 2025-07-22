import os
import logging
import chromadb
from chromadb.config import Settings
from typing import List, Dict
import PyPDF2
from dotenv import load_dotenv
from datetime import datetime
import uuid

load_dotenv()

logger = logging.getLogger(__name__)

class ChromaManager:
    def __init__(self, persist_directory: str = "chromadb"):
        """Initialize ChromaDB client and collections"""
        try:
            # Configure OpenAI API
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )
            )
            
            # Initialize embedding function
            self.embedding_function = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-3-small"
            )
            
            # Initialize collections
            self._init_knowledge_collection()
            self._init_conversation_collection()
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            self._fallback_initialization()

    def _init_knowledge_collection(self):
        """Initialize knowledge base collection"""
        try:
            self.knowledge_collection = self.client.get_collection(
                name="counseling_knowledge"
            )
            logger.info(f"Retrieved existing knowledge collection: {self.knowledge_collection.name}")
        except Exception:
            self.knowledge_collection = self.client.create_collection(
                name="counseling_knowledge",
                metadata={"description": "Couple counseling knowledge base"},
                embedding_function=self.embedding_function
            )
            logger.info(f"Created new knowledge collection: {self.knowledge_collection.name}")

    def _init_conversation_collection(self):
        """Initialize conversation history collection"""
        try:
            self.conversation_collection = self.client.get_collection(
                name="client_conversations"
            )
            logger.info(f"Retrieved existing conversation collection: {self.conversation_collection.name}")
        except Exception:
            self.conversation_collection = self.client.create_collection(
                name="client_conversations",
                metadata={"description": "Client conversation history"},
                embedding_function=self.embedding_function
            )
            logger.info(f"Created new conversation collection: {self.conversation_collection.name}")

    def _fallback_initialization(self):
        """Fallback initialization without OpenAI embeddings"""
        try:
            logger.warning("Attempting fallback initialization with default embeddings")
            self.knowledge_collection = self.client.get_or_create_collection(
                name="counseling_knowledge_fallback",
                metadata={"description": "Couple counseling knowledge base (fallback)"}
            )
            self.conversation_collection = self.client.get_or_create_collection(
                name="client_conversations_fallback",
                metadata={"description": "Client conversation history (fallback)"}
            )
            logger.info("Fallback ChromaDB initialized successfully")
        except Exception as fallback_error:
            logger.error(f"Fallback initialization also failed: {str(fallback_error)}")
            raise

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            raise

    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading TXT file {file_path}: {str(e)}")
            raise

    def ingest_document(self, file_path: str, metadata: Dict = None) -> bool:
        """Ingest a document (PDF or TXT) into ChromaDB knowledge base"""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            filename = os.path.basename(file_path)
            
            # Extract text based on file type
            if file_extension == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_extension == '.txt':
                text = self.extract_text_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Split text into chunks
            chunks = self.chunk_text(text)
            
            if not chunks:
                logger.warning(f"No content extracted from {filename}")
                return False
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            base_metadata = {
                "filename": filename,
                "file_type": file_extension,
                "total_chunks": len(chunks),
                "content_type": "knowledge_base"
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "chunk_size": len(chunk)
                })
                
                documents.append(chunk)
                metadatas.append(chunk_metadata)
                ids.append(f"{filename}_chunk_{i}")
            
            # Add to ChromaDB knowledge collection
            self.knowledge_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully ingested {filename} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest document {file_path}: {str(e)}")
            return False

    def store_conversation_chunk(self, client_id: int, messages: List[Dict], session_id: str = None) -> bool:
        """Store a chunk of conversation history"""
        try:
            if not messages:
                return False
            
            # Create conversation text from messages
            conversation_text = ""
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                conversation_text += f"{role}: {content}\n"
            
            # Generate unique ID for this conversation chunk
            chunk_id = f"client_{client_id}_conv_{uuid.uuid4().hex[:8]}"
            
            # Prepare metadata
            metadata = {
                "client_id": client_id,
                "content_type": "conversation",
                "timestamp": datetime.now().isoformat(),
                "message_count": len(messages),
                "session_id": session_id or "default",
                "chunk_size": len(conversation_text)
            }
            
            # Store in conversation collection
            self.conversation_collection.add(
                documents=[conversation_text],
                metadatas=[metadata],
                ids=[chunk_id]
            )
            
            logger.info(f"Stored conversation chunk for client {client_id} with {len(messages)} messages")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store conversation chunk for client {client_id}: {str(e)}")
            return False

    def search_conversation_history(self, client_id: int, query: str, n_results: int = 3) -> List[Dict]:
        """Search conversation history for a specific client"""
        try:
            results = self.conversation_collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"client_id": client_id},
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results['distances'] else 0.0,
                        "source": "conversation_history"
                    })
            
            logger.info(f"Found {len(formatted_results)} conversation history results for client {client_id}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Conversation history search error for client {client_id}: {str(e)}")
            return []

    def search_knowledge_base(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search knowledge base"""
        try:
            results = self.knowledge_collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results['distances'] else 0.0,
                        "source": "knowledge_base"
                    })
            
            logger.info(f"Found {len(formatted_results)} knowledge base results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Knowledge base search error: {str(e)}")
            return []

    def hybrid_search(self, client_id: int, query: str, knowledge_results: int = 2, conversation_results: int = 2) -> Dict:
        """Perform hybrid search across both knowledge base and conversation history"""
        try:
            # Search both collections
            knowledge_results = self.search_knowledge_base(query, knowledge_results)
            conversation_results = self.search_conversation_history(client_id, query, conversation_results)
            
            return {
                "knowledge_base": knowledge_results,
                "conversation_history": conversation_results,
                "total_results": len(knowledge_results) + len(conversation_results)
            }
            
        except Exception as e:
            logger.error(f"Hybrid search error for client {client_id}: {str(e)}")
            return {"knowledge_base": [], "conversation_history": [], "total_results": 0}

    # Legacy search method for backward compatibility
    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        """Legacy search method - searches knowledge base only"""
        return self.search_knowledge_base(query, n_results)

    def get_collection_stats(self) -> Dict:
        """Get statistics about both collections"""
        try:
            knowledge_count = self.knowledge_collection.count()
            conversation_count = self.conversation_collection.count()
            
            return {
                "knowledge_base_documents": knowledge_count,
                "conversation_chunks": conversation_count,
                "total_documents": knowledge_count + conversation_count,
                "collections": {
                    "knowledge": self.knowledge_collection.name,
                    "conversations": self.conversation_collection.name
                }
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}

    def delete_documents_by_filename(self, filename: str) -> bool:
        """Delete all chunks of a specific file from knowledge base"""
        try:
            # Get all documents with the filename
            results = self.knowledge_collection.get(
                where={"filename": filename},
                include=["metadatas"]
            )
            
            if results['ids']:
                self.knowledge_collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for file: {filename}")
                return True
            else:
                logger.warning(f"No documents found for filename: {filename}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting documents for {filename}: {str(e)}")
            return False

    def delete_client_conversations(self, client_id: int) -> bool:
        """Delete all conversation history for a specific client"""
        try:
            results = self.conversation_collection.get(
                where={"client_id": client_id},
                include=["metadatas"]
            )
            
            if results['ids']:
                self.conversation_collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} conversation chunks for client {client_id}")
                return True
            else:
                logger.warning(f"No conversation history found for client {client_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting conversation history for client {client_id}: {str(e)}")
            return False