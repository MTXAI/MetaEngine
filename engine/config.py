import os.path
from pathlib import Path

from engine.utils.config import EasyConfig

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
WORKSPACE_ROOT = PROJECT_ROOT / ".workspace"

APP_LOG_PATH = WORKSPACE_ROOT / "logs"
os.makedirs(APP_LOG_PATH, exist_ok=True)
APP_LOG_FILE = APP_LOG_PATH / "app.log"

DATABASE_PATH = WORKSPACE_ROOT / "data/file-rag-chroma_db"
os.makedirs(DATABASE_PATH, exist_ok=True)


class LLMModelConfig(EasyConfig):
    model_id: str
    api_key: str
    api_base_url: str
    def __init__(self, d):
        super().__init__(d)


QWEN_LLM_MODEL = LLMModelConfig(
    dict(
        model_id="qwen2.5-vl-72b-instruct",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        api_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
)

ONE_API_LLM_MODEL = LLMModelConfig(
    dict(
        model_id="qwen-turbo",
        api_key="sk-A3DJFMPvXa7Ot9faF4882708Aa2b419c87A50fFe8223B297",
        api_base_url="http://localhost:3000/v1",
    )
)
