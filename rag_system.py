import os
import logging
from typing import List, Dict, Any
from pathlib import Path
import re

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import time

class RAGSystem:
    """
    A RAG (Retrieval-Augmented Generation) system that works in a local environment using Ollama.

    Attributes:
        embedding_model (str): The name of the embedding model to use.
        llm_model (str): The name of the language model to use.
        chunk_size (int): The size of document chunks.
        chunk_overlap (int): The overlap between chunks.
        vector_db_path (str): The path to save the vector database.
        search_k (int): The number of top documents to retrieve during search.
        embeddings (OllamaEmbeddings): The Ollama embedding model for embedding documents.
        text_splitter (RecursiveCharacterTextSplitter): The text splitter for chunking text.
        llm (OllamaLLM): The Ollama language model for text generation.
        prompt (ChatPromptTemplate): The template for the prompt to be passed to the LLM.
        _rag_chain (Runnable): The RAG execution chain.
    """
    def __init__(
        self, 
        embedding_model: str = 'mxbai-embed-large',
        llm_model: str = 'gemma2:9b',
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        vector_db_path: str = './vector_db',
        search_k: int = 5,
    ):
        """
        Initializes the RAG system.

        Args:
            embedding_model (str, optional): The embedding model to use. Defaults to 'mxbai-embed-large'.
            llm_model (str, optional): The language model to use. Defaults to 'gemma2:9b'.
            chunk_size (int, optional): The chunk size for documents. Defaults to 512.
            chunk_overlap (int, optional): The overlap between chunks. Defaults to 100.
            vector_db_path (str, optional): The path to save the vector database. Defaults to './vector_db'.
            search_k (int, optional): The number of top documents to retrieve during search. Defaults to 5.
        """
        # Check Ollama installation
        self._check_ollama_installation()

        # Configure embedding model
        self.embeddings = OllamaEmbeddings(model=embedding_model)

        # Configure text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )

        # Set vector database path
        self.vector_db_path = vector_db_path

        # Configure LLM
        self.llm = OllamaLLM(model=llm_model)

        # Define prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful and intelligent assistant. Use the following context to answer the question.
            Only use information from the context, and if you don't know, answer "I do not have enough information."

            Context: {context}

            Question: {question}

            Answer in detail, citing your sources. Please specify the information of the documents that you have referenced.
            """
        )

        # Initialize RAG chain
        self._rag_chain = None
        self.search_k = search_k

    def _check_ollama_installation(self):
        """
        Checks if Ollama is installed and functioning correctly.

        Raises:
            RuntimeError: If Ollama is not installed or cannot be executed.
        """
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("Ollama is not installed or cannot be executed correctly.")
            logging.info("Ollama environment confirmed.")
        except FileNotFoundError:
            raise RuntimeError("Ollama is not installed. Please install from https://ollama.com.")

    def _load_document(self, file_path: str) -> List[Document]:
        """
        Loads a document based on its file format.

        Args:
            file_path (str): The path to the document.

        Returns:
            List[Document]: A list of loaded documents.

        Raises:
            ValueError: If the file format is not supported.
            Exception: If there is an error loading the document.
        """
        ext = os.path.splitext(file_path)[1].lower()

        loaders = {
            '.pdf': PyMuPDFLoader,
            '.txt': TextLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.pptx': UnstructuredPowerPointLoader
        }
        
        loader_class = loaders.get(ext)
        
        if not loader_class:
            raise ValueError(f"Unsupported file format: {ext}")
        
        try:
            loader = loader_class(file_path)
            documents = loader.load()
            
            # Add/Ensure metadata fields
            for i, doc in enumerate(documents):
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = file_path  # ファイルパス
                if 'page' not in doc.metadata:
                    doc.metadata['page'] = i + 1 # ページ番号
                # Dummy data for title, summary, and keywords (Replace with actual logic if possible)
                if 'title' not in doc.metadata:
                    doc.metadata['title'] = os.path.basename(file_path) # ファイル名
                if 'summary' not in doc.metadata:
                     doc.metadata['summary'] =  doc.page_content[:200] if len(doc.page_content) > 200 else doc.page_content # 最初の200文字
                if 'keywords' not in doc.metadata:
                     words = re.findall(r'\b\w+\b', doc.page_content.lower())
                     word_counts = {}
                     for word in words:
                       word_counts[word] = word_counts.get(word, 0) + 1
                     sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
                     doc.metadata['keywords'] = ', '.join(sorted_words[:5]) #上位5つの単語
            return documents
        except Exception as e:
            logging.error(f"Error loading document: {e} ({file_path})")
            raise

    def process_documents(self, document_paths: List[str], additional_document_paths: List[str] = None):
        """
        Processes the specified document paths and builds a vector database.

        This function handles both directory paths and individual file paths. It also includes
        an option to process additional documents passed separately.

        Args:
            document_paths (List[str]): A list of document paths to process. It can include paths
                to directories containing PDF files or individual file paths.
            additional_document_paths (List[str], optional): An optional list of additional document paths
                to process. Defaults to None.
        """
        all_splits = []
        start_time = time.time()

        # Process PDF files within specified directories
        for path_str in document_paths:
             path = Path(path_str)
             if path.is_dir():
                for file_path in path.glob("*.pdf"):
                     try:
                        documents = self._load_document(str(file_path))
                        splits = self.text_splitter.split_documents(documents)
                        all_splits.extend(splits)
                        logging.info(f"Processed document: {file_path}")
                     except Exception as e:
                         logging.error(f"Error processing document: {e} ({file_path})")
                         continue
             else:
                try:
                    documents = self._load_document(path_str)
                    splits = self.text_splitter.split_documents(documents)
                    all_splits.extend(splits)
                    logging.info(f"Processed document: {path_str}")
                except Exception as e:
                    logging.error(f"Error processing document: {e} ({path_str})")
                    continue

        # Process additional documents
        if additional_document_paths:
          for path in additional_document_paths:
            try:
                documents = self._load_document(path)
                splits = self.text_splitter.split_documents(documents)
                all_splits.extend(splits)
                logging.info(f"Processed additional document: {path}")
            except Exception as e:
                logging.error(f"Error processing additional document: {e} ({path})")
                continue
        
        if not all_splits:
            logging.warning("No document chunks to process.")
            return

        try:
            # Create and save the vector store
            vectorstore = FAISS.from_documents(all_splits, self.embeddings)
            
            # Save the vector store securely
            vectorstore.save_local(self.vector_db_path)
            
            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"Processed {len(all_splits)} document chunks ({duration:.2f} seconds). Vector store saved to: {self.vector_db_path}")
        except Exception as e:
            logging.error(f"Error creating or saving vector store: {e}")
    
    def _extract_metadata(self, doc_metadata: Dict[str, Any]) -> str:
        """
        Extracts necessary information from document metadata.

        Args:
            doc_metadata (Dict[str, Any]): The metadata of the document.

        Returns:
            str: Extracted metadata information as a string.
        """
        # Example: Extract source, title, author, page, summary, and keywords
        extracted_info = ""
        if "source" in doc_metadata:
            extracted_info += f"Source: {doc_metadata['source']}\n"
        if "title" in doc_metadata:
            extracted_info += f"Title: {doc_metadata['title']}\n"
        if "author" in doc_metadata:
            extracted_info += f"Author: {doc_metadata['author']}\n"
        if "page" in doc_metadata:
            extracted_info += f"Page: {doc_metadata['page']}\n"
        if "summary" in doc_metadata:
            extracted_info += f"Summary: {doc_metadata['summary']}\n"
        if "keywords" in doc_metadata:
            extracted_info += f"Keywords: {doc_metadata['keywords']}\n"
        
        return extracted_info.strip()  # Remove unnecessary newlines

    def _create_rag_chain(self, vectorstore):
        """
        Creates the RAG (Retrieval-Augmented Generation) chain.

        Args:
            vectorstore: The FAISS vector store object.
        """
        retriever = vectorstore.as_retriever(search_kwargs={"k": self.search_k})
        
        def format_docs(docs):
            formatted_docs = []
            for doc in docs:
                extracted_metadata = self._extract_metadata(doc.metadata)
                formatted_docs.append(f"Document: {doc.page_content}\nMetadata: {extracted_metadata}")
            return "\n\n".join(formatted_docs)

        
        self._rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, query: str) -> Dict[str, Any]:
        """
        Generates a response to a query.

        Args:
            query (str): The user's question.

        Returns:
            Dict[str, Any]: A dictionary containing the response, retrieved documents, and processing time.
        """
        start_time = time.time()
        # Securely load the vector store
        try:
            vectorstore = FAISS.load_local(
                self.vector_db_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logging.error(f"Error loading vector store: {e}")
            raise

        # Create the RAG chain
        self._create_rag_chain(vectorstore)
        
        # Generate the response
        try:
            response = self._rag_chain.invoke(query)
        except Exception as e:
            logging.error(f"LLM response error: {e}")
            return {'error': f'LLM response error: {e}'}

        
        # Get relevant documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": self.search_k})
        relevant_docs = retriever.invoke(query)

        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Query processing completed in {duration:.2f} seconds.")
        
        return {
            'response': response,
            'retrieved_docs': [
                {
                    'content': doc.page_content,
                    'metadata': self._extract_metadata(doc.metadata)
                } for doc in relevant_docs
            ]
        }
    
