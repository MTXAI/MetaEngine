import logging
import threading
from contextlib import contextmanager


class RWLock:
    """
    读写锁实现，允许多个读操作同时进行，但写操作是互斥的。
    基于 Python 的 RLock 实现，模仿 Go 语言的 sync.RWMutex 接口。
    """

    def __init__(self):
        self._rw_mutex = threading.RLock()  # 保护内部状态的重入锁
        self._readers = 0  # 当前读锁持有者数量
        self._writer_active = False  # 是否有活跃的写锁
        self._writer_waiting = 0  # 等待的写锁数量（用于公平性）

        # 条件变量：用于等待读锁和写锁
        self._read_cond = threading.Condition(self._rw_mutex)
        self._write_cond = threading.Condition(self._rw_mutex)

    def RLock(self):
        """获取读锁（对应 Go 的 RLock）"""
        with self._rw_mutex:
            # 等待没有活跃的写锁且没有等待的写锁（可选的公平性策略）
            while self._writer_active or (self._writer_waiting > 0 and self._readers > 0):
                self._read_cond.wait()
            self._readers += 1

    def RUnlock(self):
        """释放读锁（对应 Go 的 RUnlock）"""
        with self._rw_mutex:
            if self._readers <= 0:
                raise RuntimeError("RUnlock 调用没有匹配的 RLock")
            self._readers -= 1

            # 如果没有读锁了，通知等待的写锁
            if self._readers == 0:
                self._write_cond.notify_all()

    def Lock(self):
        """获取写锁（对应 Go 的 Lock）"""
        with self._rw_mutex:
            # 增加等待的写锁计数
            self._writer_waiting += 1

            # 等待没有活跃的读锁和写锁
            while self._readers > 0 or self._writer_active:
                self._write_cond.wait()

            # 减少等待计数，标记为活跃的写锁
            self._writer_waiting -= 1
            self._writer_active = True

    def Unlock(self):
        """释放写锁（对应 Go 的 Unlock）"""
        with self._rw_mutex:
            if not self._writer_active:
                raise RuntimeError("Unlock 调用没有匹配的 Lock")

            # 标记写锁不再活跃，并通知所有等待的读锁和写锁
            self._writer_active = False
            self._write_cond.notify_all()
            self._read_cond.notify_all()

    # 上下文管理器支持（可选，方便使用 with 语句）
    @contextmanager
    def reader_lock(self):
        """读锁的上下文管理器"""
        try:
            self.RLock()
            yield
        finally:
            self.RUnlock()

    @contextmanager
    def writer_lock(self):
        """写锁的上下文管理器"""
        try:
            self.Lock()
            yield
        finally:
            self.Unlock()

    # 状态检查方法（用于调试）
    def status(self):
        """返回当前锁的状态（用于调试）"""
        with self._rw_mutex:
            return {
                "readers": self._readers,
                "writer_active": self._writer_active,
                "writer_waiting": self._writer_waiting
            }


if __name__ == '__main__':
    import threading
    import time

    # 创建读写锁实例
    rw_lock = RWLock()
    shared_data = 0


    def reader(id):
        global shared_data
        while True:
            with rw_lock.reader_lock():
                logging.info(f"Reader {id} got read lock, data: {shared_data}")
                time.sleep(0.1)
            time.sleep(0.2)  # 释放锁后稍等


    def writer(id):
        global shared_data
        while True:
            with rw_lock.writer_lock():
                shared_data += 1
                logging.info(f"Writer {id} updated data to {shared_data}")
                time.sleep(0.2)
            time.sleep(0.5)  # 释放锁后稍等


    # 创建并启动线程
    readers = [threading.Thread(target=reader, args=(i,)) for i in range(3)]
    writers = [threading.Thread(target=writer, args=(i,)) for i in range(1)]

    for t in readers + writers:
        t.daemon = True
        t.start()

    # 主线程保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Exiting...")
