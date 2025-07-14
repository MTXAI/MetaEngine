class BaseProcessor:
    """
    处理器基类，定义输入输出预处理和后处理接口。
    """
    def preprocess(self, text: str) -> str:
        """
        输入预处理，可自定义实现。
        """
        return text

    def postprocess(self, text: str) -> str:
        """
        输出后处理，可自定义实现。
        """
        return text

class SimpleProcessor(BaseProcessor):
    """
    示例：简单处理器，去除多余空格。
    """
    def preprocess(self, text: str) -> str:
        return text.strip()

    def postprocess(self, text: str) -> str:
        return text.strip()

