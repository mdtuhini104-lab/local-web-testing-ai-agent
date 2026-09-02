"""
Self-Learning Engine module using DuckDuckGo Search and ChromaDB.
Allows the agent to search the internet for testing patterns and store them in a local vector database.
"""
import logging
from typing import List, Dict, Any, Optional
from duckduckgo_search import DDGS
from app.config import settings

logger = logging.getLogger("learning_engine")
logging.basicConfig(level=logging.INFO)

# Monkey-patch tokenizer issues if transformers is loaded
try:
    from transformers import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens") or isinstance(getattr(PreTrainedTokenizerBase, "additional_special_tokens", None), list):
        PreTrainedTokenizerBase.additional_special_tokens = property(
            lambda self: getattr(self, "_additional_special_tokens", []) or self.special_tokens_map.get("additional_special_tokens", [])
        )
except Exception:
    pass

class LearningEngine:
    def __init__(self):
        self.collection = None
        self.in_memory_docs: List[Dict[str, str]] = []
        self.ddgs = DDGS()
        
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            
            self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            self.collection = self.chroma_client.get_or_create_collection(
                name="qa_knowledge_base",
                embedding_function=self.embedding_fn
            )
            logger.info("✅ ChromaDB Vector Knowledge Store initialized successfully.")
        except Exception as e:
            logger.warning(f"⚠️ ChromaDB embedding initialization failed: {e}. Falling back to in-memory store.")

    def search_and_learn(self, query: str) -> List[str]:
        """
        Searches the internet for the given query, extracts snippets,
        and saves them into the local knowledge base.
        """
        logger.info(f"🔍 Searching Internet for: {query}")
        try:
            results = list(self.ddgs.text(query, max_results=5))
            learned_snippets = []
            
            for i, res in enumerate(results):
                title = res.get('title', '')
                snippet = res.get('body', '')
                url = res.get('href', '')
                
                content = f"Title: {title}\nSnippet: {snippet}\nURL: {url}"
                doc_id = f"doc_{hash(url + snippet)}"
                
                if self.collection:
                    try:
                        existing = self.collection.get(ids=[doc_id])
                        if not existing['ids']:
                            self.collection.add(
                                documents=[content],
                                metadatas=[{"url": url, "title": title, "query": query}],
                                ids=[doc_id]
                            )
                            learned_snippets.append(content)
                            logger.info(f"🧠 Learned new knowledge from: {url}")
                    except Exception as embed_err:
                        logger.warning(f"⚠️ Failed to add document to vector store: {embed_err}")
                        self.in_memory_docs.append({"content": content, "query": query})
                        learned_snippets.append(content)
                else:
                    self.in_memory_docs.append({"content": content, "query": query})
                    learned_snippets.append(content)
                    logger.info(f"🧠 Learned new knowledge (in-memory): {url}")
            
            return learned_snippets
        except Exception as e:
            logger.error(f"❌ Failed to search internet: {e}")
            return []

    def retrieve_context(self, query: str, n_results: int = 3) -> str:
        """
        Retrieves relevant knowledge from the local vector database or in-memory fallback.
        """
        try:
            if self.collection:
                try:
                    if self.collection.count() == 0:
                        return ""
                    
                    results = self.collection.query(
                        query_texts=[query],
                        n_results=min(n_results, self.collection.count())
                    )
                    
                    if not results['documents'] or not results['documents'][0]:
                        return ""
                    
                    return "\n\n".join(results['documents'][0])
                except Exception as query_err:
                    logger.warning(f"⚠️ Vector search query failed: {query_err}. Falling back to in-memory search.")
            
            # In-memory fallback
            if not self.in_memory_docs:
                return ""
            
            matching = [doc["content"] for doc in self.in_memory_docs[:n_results]]
            return "\n\n".join(matching)
        except Exception as e:
            logger.error(f"❌ Failed to retrieve context: {e}")
            return ""
