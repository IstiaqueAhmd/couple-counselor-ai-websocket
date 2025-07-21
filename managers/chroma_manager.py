import os
import logging
import chromadb
from chromadb.config import Settings
from typing import List, Dict
import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class ChromaManager:
    def __init__(self, persist_directory: str = "chromadb"):
        """Initialize ChromaDB client and collection"""
        try:
            # Configure Gemini API
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(allow_reset=True)
            )
            
            # Get or create collection for counseling knowledge with Gemini embeddings
            self.collection = self.client.get_or_create_collection(
                name="counseling_knowledge",
                metadata={"description": "Couple counseling knowledge base"},
                embedding_function=chromadb.utils.embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                    api_key=api_key,
                    model_name="models/text-embedding-004"
                )
            )
            
            logger.info(f"ChromaDB initialized with collection: {self.collection.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
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
        """Ingest a document (PDF or TXT) into ChromaDB"""
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
                "total_chunks": len(chunks)
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
            
            # Add to ChromaDB collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully ingested {filename} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest document {file_path}: {str(e)}")
            return False

    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        try:
            results = self.collection.query(
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
                        "distance": results['distances'][0][i] if results['distances'] else 0.0
                    })
            
            logger.info(f"Found {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection.name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}

    def delete_documents_by_filename(self, filename: str) -> bool:
        """Delete all chunks of a specific file"""
        try:
            # Get all documents with the filename
            results = self.collection.get(
                where={"filename": filename},
                include=["metadatas"]
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for file: {filename}")
                return True
            else:
                logger.warning(f"No documents found for filename: {filename}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting documents for {filename}: {str(e)}")
            return False