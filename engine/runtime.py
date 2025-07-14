import asyncio
import os

from engine.config import DEFAULT_RUNTIME_CONFIG
from engine.utils.concurrent.pool import ThreadPool


main_loop = asyncio.new_event_loop()
asyncio.set_event_loop(main_loop)

thread_pool = ThreadPool(
    max_workers=DEFAULT_RUNTIME_CONFIG.max_workers,
    max_queue_size=DEFAULT_RUNTIME_CONFIG.max_queue_size,
)

def set_environment_variables():
    # disable tokenizer fork
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_coroutine_threadsafe(coro):
    asyncio.run_coroutine_threadsafe(coro, loop=main_loop)

def run_forever():
    main_loop.run_forever()

def run_until_complete(coro):
    main_loop.run_until_complete(coro)

