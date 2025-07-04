from engine.config import DEFAULT_PROJECT_CONFIG
from engine.runtime import set_environment_variables
from engine.utils.logging import init_logging

set_environment_variables()
init_logging(DEFAULT_PROJECT_CONFIG.app_log_file)
