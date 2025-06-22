import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from transformers import AutoTokenizer

from core.config.config import DATABASE_PATH
from engine.agent.file_rag.file_loader import is_supported_file, load_file
from core.utils.log import logger

def load_db():
    """
    Load the ChromaDB vector store from the specified database path.
    If the database does not exist, it will return None.
    """
    if os.path.exists(DATABASE_PATH):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"Loading database from {DATABASE_PATH}")
        vector_store = Chroma(embedding_function=embeddings, persist_directory=DATABASE_PATH)
        return vector_store
    else:
        logger.warning(f"Database {DATABASE_PATH} does not exist.")
        return None

def create_db(dir : str):
    # load file in dir
    source_docs = []
    for file in os.listdir(dir):
        if is_supported_file(file):
            file_path = os.path.join(dir, file)
            logger.info(f"Loading file: {file_path}")
            source_docs.append(load_file(file_path))

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained("thenlper/gte-small"),
        chunk_size=200,
        chunk_overlap=20,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    # Split docs and keep only unique ones
    logger.info("正在拆分文档...")
    docs_processed = []
    ids = []
    unique_texts = set()
    for doc in source_docs:
        new_docs = text_splitter.split_documents(doc)
        new_docs = filter_complex_metadata(new_docs)
        for new_doc in new_docs:
            if new_doc.page_content not in unique_texts:
                unique_texts.add(new_doc.page_content)
                if "id" not in new_doc.metadata:
                    ids.append(str(len(docs_processed)))
                docs_processed.append(new_doc)

    print("Embedding documents... This should take a few minutes (5 minutes on MacBook with M1 Pro)")
    # Initialize embeddings and ChromaDB vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(documents=docs_processed, ids=ids, embedding=embeddings, persist_directory=DATABASE_PATH)

    return vector_store

def clean_db():
    if os.path.exists(DATABASE_PATH):
        import shutil
        shutil.rmtree(DATABASE_PATH)
        logger.info(f"Database {DATABASE_PATH} has been cleaned.")
    else:
        logger.warning(f"Database {DATABASE_PATH} does not exist.")