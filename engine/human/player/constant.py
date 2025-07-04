

# 数字人状态控制与转移:
#   待机 ready
#   接收到文字 ready -> speaking -> ready
#   中止动作 speaking -> interrupt -> ready
StateReady = 0  # 正在待机, 可以正常响应任何请求, 按照 fps 定时产生空白音频帧
StateBusy = 1  # 忙碌中, 此时中断其他动作(正在进行的暂时不能中断), 并且需要清空尚未播放的帧, 回归待机, 清空过程中不接受其他动作
