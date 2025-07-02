from engine.config import DEFAULT_PROJECT_CONFIG
from engine.utils.logging import init_logging

init_logging(DEFAULT_PROJECT_CONFIG.app_log_file)
