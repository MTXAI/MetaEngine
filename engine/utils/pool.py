import atexit
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Full

from engine.utils.common import EasyDict


class TaskCallback:
    def on_submit(self, future, task_info):
        pass

    def on_schedule(self, future, task_info):
        pass

    def on_complete(self, future, task_info):
        pass

class TaskInfo(EasyDict):
    name: str
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

class ThreadPool:
    def __init__(self, max_workers, max_queue_size):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._task_queue = Queue(maxsize=max_queue_size)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._status_lock = threading.Lock()
        self._status = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "running_tasks": 0,
            "scheduled_tasks": 0,
            "failed_tasks": 0,
        }
        self._task_metadata = {}
        threading.Thread(target=self._monitor_daemon, daemon=True).start()
        atexit.register(self.shutdown)

    def _update_status(self, shutdown=False):
        with self._status_lock:
            self._status["running_tasks"] = len([fut for fut in self._task_queue.queue if not fut.done()])
            if shutdown:
                self._status["completed_tasks"] = self._status["total_tasks"] - self._status["running_tasks"]
                self._status["scheduled_tasks"] = self._task_queue.qsize()

    def _monitor_daemon(self):
        while not self._stop_event.is_set():
            self._update_status()
            time.sleep(5)

    def _task_done(self, future, task_info, callback:TaskCallback=None):
        with self._lock:
            self._task_queue.get_nowait()
            self._task_queue.task_done()
        if future.exception() is not None:
            with self._status_lock:
                self._status["failed_tasks"] += 1
        if callback:
            callback.on_complete(future, task_info)
        with self._status_lock:
            self._status["completed_tasks"] += 1
            self._status["scheduled_tasks"] -= 1

    def submit(self, fn, *args, task_info: TaskInfo=None, callback: TaskCallback=None, **kwargs):
        if task_info is None:
            task_info = TaskInfo('')
        while not self._stop_event.is_set():
            try:
                future = self.executor.submit(fn, *args, **kwargs)
                if callback:
                    callback.on_submit(future, task_info)
                with self._lock:
                    self._task_metadata[id(future)] = task_info
                self._task_queue.put(future, block=False)  # Non-blocking put
                if callback:
                    callback.on_schedule(future, task_info)
                future.add_done_callback(lambda fut: self._task_done(fut, task_info, callback))
                with self._status_lock:
                    self._status["total_tasks"] += 1
                    self._status["scheduled_tasks"] += 1
                return future
            except Full:
                logging.warning("Task queue is full, waiting...")
                time.sleep(1)  # Wait for a second before retrying
        return None

    def map(self, fn, iterable, timeout=None):
        futures = [self.submit(fn, item) for item in iterable]
        return as_completed(futures, timeout=timeout)

    def get_status(self):
        with self._status_lock:
            return self._status.copy()

    def wait_for_task(self, future):
        with self._lock:
            task_info = self._task_metadata[id(future)]
        try:
            result = future.result()
            return result
        except Exception as e:
            logging.warning(f"Task failed with exception: {e}, info: {task_info}")
            return None

    def shutdown(self, wait=True, cancel=False):
        self._stop_event.set()
        self.executor.shutdown(wait=wait, cancel_futures=cancel)
        self._update_status()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()


if __name__ == "__main__":
    def example_task(i):
        def fn(n):
            time.sleep(1)  # Simulate some work
            if random.random() > 0.5:
                raise Exception(f'random error: #{i}')
            return n * n

        return fn


    class SimpleCallback(TaskCallback):
        def on_submit(self, future, task_info):
            print(f"Task submitted: {future}, info: {task_info}")

        def on_schedule(self, future, task_info):
            print(f"Task scheduled: {future}, info: {task_info}")

        def on_complete(self, future, task_info):
            try:
                result = future.result()
                print(f"Task completed, Result: {result}, info: {task_info}")
            except Exception as e:
                print(f"Task completed, Error: {e}, info: {task_info}")


    with ThreadPool(max_workers=4, max_queue_size=5) as pool:
        try:
            submitted_futures = []
            for i in range(20):  # Submit more tasks than the queue can hold initially
                task_info = TaskInfo(
                    name="Task #{}".format(i),
                )
                future = pool.submit(
                    example_task(i),
                    i,
                    task_info=task_info,
                    callback=SimpleCallback(),
                )
                print(f"Submitted task {i}")
                submitted_futures.append(future)
        except Exception as e:
            print(e)

        # Get and print results for each submitted task
        for future in submitted_futures:
            result = pool.wait_for_task(future)
            print(f"Result for task {submitted_futures.index(future)}: {result}")

        pool.shutdown(wait=True)

        # Get and print current status
        status = pool.get_status()
        print("Current Status:", status)
