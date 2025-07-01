import os.path
from pathlib import Path

from engine.utils.common import get_device_and_start_method
from engine.utils.config import EasyConfig


class ProjectConfig(EasyConfig):
    root_path: Path
    workspace_path: Path
    app_log_path: Path
    app_log_file: Path
    vecdb_path: Path
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        os.makedirs(self.app_log_path, exist_ok=True)

_default_root_path: Path = Path(__file__).parent.parent.absolute()
_default_workspace_path: Path = _default_root_path / '.workspace'
_default_app_log_path: Path = _default_workspace_path / 'logs'
_default_app_log_file: Path = _default_app_log_path / 'app.log'
_default_vecdb_path: Path = _default_workspace_path / 'data/file-rag-chroma_db'
DEFAULT_PROJECT_CONFIG = ProjectConfig(
    dict(
        root_path=_default_root_path,
        workspace_path=_default_workspace_path,
        app_log_path=_default_app_log_path,
        app_log_file=_default_app_log_file,
        vecdb_path=_default_vecdb_path,
    )
)


class LLMModelConfig(EasyConfig):
    model_id: str
    api_key: str
    api_base_url: str
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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

_device, start_method = get_device_and_start_method()
class RuntimeConfig(EasyConfig):
    device: str
    start_method: str
    max_workers: int
    max_queue_size: int
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = _device
        self.start_method = start_method

DEFAULT_RUNTIME_CONFIG = RuntimeConfig(
    dict(
        max_workers=8,
        max_queue_size=12,
    )
)


class PlayerConfig(EasyConfig):
    fps: int
    sample_rate: int
    batch_size: int
    window_left: int
    window_right: int
    warmup_iters: int
    timeout: float
    audio_ptime: float
    video_ptime: float
    video_clock_rate: int
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_iters = self.window_left + self.window_right

WAV2LIP_PLAYER_CONFIG = PlayerConfig(
    dict(
        fps=90,
        sample_rate=16000,
        batch_size=16,
        window_left=10,
        window_right=10,
        timeout=1 / float(50) / 2,
        audio_ptime=0.02,
        video_ptime=0.04,
        video_clock_rate=90000,
    )
)

