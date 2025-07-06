import asyncio


class ObservableQueue:
    """
    包装 asyncio.Queue，提供等待队列有数据但不读取的功能
    """

    def __init__(self, maxsize: int = 0):
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.condition = asyncio.Condition()

    def clear(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()  # 如果使用了 join()，需要标记任务完成
            except asyncio.QueueEmpty:
                break

    async def put(self, item):
        """向队列中添加元素，并通知等待者"""
        await self.queue.put(item)
        async with self.condition:
            self.condition.notify_all()

    async def get(self):
        """从队列中获取元素（与原生Queue行为一致）"""
        return await self.queue.get()

    async def wait_for_data(self):
        """等待队列中有数据，但不读取数据"""
        async with self.condition:
            while self.queue.empty():
                await self.condition.wait()
        # 此时队列中至少有一个元素

    def qsize(self):
        """返回队列中的元素数量"""
        return self.queue.qsize()