from typing import List

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document

from engine.agent.file_rag.chinese_text_splitter import ChineseTextSplitter


def load_file(filepath) -> List[Document]:
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredFileLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".txt"):
        loader = UnstructuredFileLoader(filepath, encoding='utf8')
        textsplitter = ChineseTextSplitter(pdf=False)
        docs = loader.load_and_split(textsplitter)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False)
        docs = loader.load_and_split(text_splitter=textsplitter)
    return docs

def is_supported_file(filename):
    """
    校验文件名是否为支持的后缀（.md, .pdf, .txt）。
    """
    return filename.lower().endswith(('.md', '.pdf', '.txt'))