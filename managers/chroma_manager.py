import os
import logging
import chromadb
from managers.db_manager import DBManager
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
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        
        # Initialize embedding function
        self.embedding_function = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )
        
        # Initialize collections
        self.knowledge_collection = self.client.get_or_create_collection(
            name="counseling_knowledge",
            metadata={"description": "Couple counseling knowledge base"},
            embedding_function=self.embedding_function
        )
        
        self.conversation_collection = self.client.get_or_create_collection(
            name="client_conversations",
            metadata={"description": "Client conversation history"},
            embedding_function=self.embedding_function
        )
        
        logger.info("ChromaDB manager initialized successfully")

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks with token limit consideration"""
        chunks = []
        start = 0
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        max_chars_per_chunk = 2000  # This should be well under the token limit
        actual_chunk_size = min(chunk_size, max_chars_per_chunk)
        
        while start < len(text):
            end = min(start + actual_chunk_size, len(text))
            chunk = text[start:end]
            
            # Break at sentence boundaries if possible
            if end < len(text):
                break_point = max(chunk.rfind('.'), chunk.rfind('\n'), chunk.rfind('!'), chunk.rfind('?'))
                if break_point > start + actual_chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    end = break_point + 1
            
            if len(chunk.strip()) > 50:
                chunks.append(chunk.strip())
            
            # Ensure we always make progress to avoid infinite loops
            next_start = end - overlap
            if next_start <= start:
                next_start = start + max(1, actual_chunk_size // 2)  # Ensure minimum progress
            
            start = next_start
            
        return chunks

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from PDF or TXT file"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return "\n".join(page.extract_text() for page in pdf_reader.pages)
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def ingest_document(self, file_path: str, metadata: Dict = None) -> bool:
        """Ingest a document into ChromaDB knowledge base"""
        try:
            filename = os.path.basename(file_path)
            text = self.extract_text_from_file(file_path)
            chunks = self.chunk_text(text)
            
            if not chunks:
                logger.warning(f"No content extracted from {filename}")
                return False
            
            logger.info(f"Processing {len(chunks)} chunks for {filename}")
            
            # Prepare data for ChromaDB
            base_metadata = {
                "filename": filename,
                "file_type": os.path.splitext(file_path)[1].lower(),
                "content_type": "knowledge_base",
                "total_chunks": len(chunks)
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            # Process chunks in smaller batches to avoid API limits
            batch_size = 50  # Process 50 chunks at a time
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(chunks))
                
                batch_chunks = chunks[start_idx:end_idx]
                batch_metadatas = [{**base_metadata, "chunk_index": i} for i in range(start_idx, end_idx)]
                batch_ids = [f"{filename}_chunk_{i}" for i in range(start_idx, end_idx)]
                
                logger.info(f"Processing batch {batch_idx + 1}/{total_batches} with {len(batch_chunks)} chunks")
                
                # Add batch to ChromaDB
                self.knowledge_collection.add(
                    documents=batch_chunks,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            
            logger.info(f"Successfully ingested {filename} with {len(chunks)} chunks in {total_batches} batches")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest document {file_path}: {str(e)}")
            return False

    def store_conversation_chunk(self, client_id: int, messages: List[Dict]) -> bool:
        """Store conversation history chunk"""
        try:
            if not messages:
                return False
            
            # Create conversation text
            conversation_text = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in messages
            ])
            
            chunk_id = f"client_{client_id}_conv_{uuid.uuid4().hex[:8]}"
            
            metadata = {
                "client_id": client_id,
                "content_type": "conversation",
                "timestamp": datetime.now().isoformat(),
                "message_count": len(messages)
            }
            
            self.conversation_collection.add(
                documents=[conversation_text],
                metadatas=[metadata],
                ids=[chunk_id]
            )
            
            logger.info(f"Stored conversation chunk for client {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store conversation chunk: {str(e)}")
            return False

    def search_knowledge_base(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search knowledge base"""
        try:
            results = self.knowledge_collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            return self._format_results(results, "knowledge_base")
            
        except Exception as e:
            logger.error(f"Knowledge base search error: {str(e)}")
            return []

    def search_conversation_history(self, client_id: int, query: str, n_results: int = 3) -> List[Dict]:
        """Search conversation history for specific client"""
        try:
            results = self.conversation_collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"client_id": client_id},
                include=["documents", "metadatas", "distances"]
            )
            
            return self._format_results(results, "conversation_history")
            
        except Exception as e:
            logger.error(f"Conversation search error: {str(e)}")
            return []

    def hybrid_search(self, client_id: int, query: str, knowledge_results: int = 2, conversation_results: int = 2) -> Dict:
        """Search both knowledge base and conversation history"""
        knowledge_data = self.search_knowledge_base(query, knowledge_results)
        conversation_data = self.search_conversation_history(client_id, query, conversation_results)
        
        # Safely get partner data
        partner_data = []
        try:
            spouse = DBManager.get_spouse(client_id)
            if spouse and "user_id" in spouse:
                partner_id = spouse["user_id"]
                partner_data = self.search_conversation_history(partner_id, query, conversation_results)
        except Exception as e:
            logger.warning(f"Could not retrieve partner data for client {client_id}: {str(e)}")

        return {
            "knowledge_base": knowledge_data,
            "conversation_history": conversation_data,
            "partner_history": partner_data,
            "total_results": len(knowledge_data) + len(conversation_data) + len(partner_data)
        }

    def _format_results(self, results, source_type: str) -> List[Dict]:
        """Format search results"""
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0.0,
                    "source": source_type
                })
        return formatted_results

    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            return {
                "knowledge_base_documents": self.knowledge_collection.count(),
                "conversation_chunks": self.conversation_collection.count()
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"error": str(e)}
