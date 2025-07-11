import os.path
from pathlib import Path

from engine.utils.common import get_device_and_start_method, check_fp16_support
from engine.utils.config import EasyConfig


# 存放各类配置, 临时方案


class ProjectConfig(EasyConfig):
    root_path: Path
    workspace_path: Path
    app_log_path: Path
    app_log_file: Path
    vecdb_path: Path
    docs_path: Path
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        os.makedirs(self.app_log_path, exist_ok=True)

_default_root_path: Path = Path(__file__).parent.parent.absolute()
_default_workspace_path: Path = _default_root_path / '.workspace'
_default_app_log_path: Path = _default_workspace_path / 'logs'
_default_app_log_file: Path = _default_app_log_path / 'app.log'
_default_vecdb_path: Path = _default_workspace_path / 'data/file-rag-chroma_db'
_default_docs_path: Path = _default_workspace_path / 'data/rag_files'
DEFAULT_PROJECT_CONFIG = ProjectConfig(
    dict(
        root_path=_default_root_path,
        workspace_path=_default_workspace_path,
        app_log_path=_default_app_log_path,
        app_log_file=_default_app_log_file,
        vecdb_path=_default_vecdb_path,
        docs_path=_default_docs_path,
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


class RuntimeConfig(EasyConfig):
    device: str
    start_method: str
    max_workers: int
    max_queue_size: int
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _device, start_method = get_device_and_start_method()
        self.device = _device
        self.start_method = start_method

        _use_fp16 = check_fp16_support()
        self.use_fp16 = _use_fp16

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
    timeout: float
    audio_ptime: float
    video_ptime: float
    frame_multiple: int
    clock_rate: int
    frame_sync_prefer: str
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

fps = 50.
frame_multiple = 2
WAV2LIP_PLAYER_CONFIG = PlayerConfig(
    dict(
        fps=int(fps),
        sample_rate=16000,
        batch_size=16,
        timeout=1/fps/2,
        audio_ptime=1/fps,
        video_ptime=1/fps*frame_multiple,
        frame_multiple=frame_multiple,
        clock_rate=90000,
        frame_sync_prefer="video",
    )
)

class VoiceProcessorConfig(EasyConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


DEFAULT_VOICE_PROCESSOR_CONFIG = VoiceProcessorConfig(
    dict()
)

class AvatarProcessorConfig(EasyConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


DEFAULT_AVATAR_PROCESSOR_CONFIG = AvatarProcessorConfig(
    dict()
)
