from typing import List

from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader

from engine.human.character.file_rag.chinese_text_splitter import ChineseTextSplitter


def load_file(filepath) -> List[Document]:
    if filepath.lower().endswith(".md"):
        loader = UnstructuredLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".txt"):
        loader = UnstructuredLoader(filepath, encoding='utf8')
        textsplitter = ChineseTextSplitter(pdf=False)
        docs = loader.load_and_split(textsplitter)
    else:
        loader = UnstructuredLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False)
        docs = loader.load_and_split(text_splitter=textsplitter)
    return docs

def is_supported_file(filename):
    """
    校验文件名是否为支持的后缀（.md, .pdf, .txt）。
    """
    return filename.lower().endswith(('.md', '.pdf', '.txt'))