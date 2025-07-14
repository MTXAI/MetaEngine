from typing import List, Union
from langchain_core.documents import Document
from engine.human.character.file_rag.file_loader import load_file

class KnowledgeBaseManager:
    """
    知识库管理器：支持自定义加载本地文件，进行分割、存储和后续检索等操作。
    """
    def __init__(self):
        self.docs: List[Document] = []

    def load(self, filepaths: Union[str, List[str]]):
        """
        加载单个或多个文件，自动分割并存储。
        :param filepaths: 文件路径或路径列表
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        for path in filepaths:
            docs = load_file(path)
            self.docs.extend(docs)

    def clear(self):
        """
        清空已加载的文档。
        """
        self.docs = []

    def get_docs(self) -> List[Document]:
        """
        获取当前所有文档。
        """
        return self.docs

    def __len__(self):
        return len(self.docs)

