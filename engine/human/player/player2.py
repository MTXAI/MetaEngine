import asyncio
import copy
import queue
import threading
import traceback
from concurrent.futures import Future
from queue import Queue
from typing import Tuple
import time

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from av import AudioFrame, VideoFrame

from engine.config import DEFAULT_RUNTIME_CONFIG, PlayerConfig
from engine.human.avatar.avatar import ModelWrapper
from engine.human.utils.data import Data
from engine.utils.pool import ThreadPool, TaskInfo

mp.set_start_method(DEFAULT_RUNTIME_CONFIG.start_method, force=True)


class Player:
    def __init__(
            self,
            config: PlayerConfig,
            model: ModelWrapper,
            avatar: Tuple,
            thread_pool: ThreadPool,
            event_loop: asyncio.AbstractEventLoop,
            # video_track: ,
            # audio_track: ,
            v_res_queue: Queue,
            a_res_queue: Queue,
    ):
        self.v_res_queue = v_res_queue
        self.a_res_queue = a_res_queue

        print(config)
        self.config = config
        self.fps = config.fps
        self.sample_rate = config.sample_rate
        self.batch_size = config.batch_size
        self.chunk_size = int(self.sample_rate / self.fps)
        self.frame_interval = config.frame_interval
        self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle = avatar
        self.model = model
        self.stop_event = threading.Event()

        # 设置队列最大大小，防止无限阻塞
        self.input_queue = Queue()
        self.frame_queue = Queue()
        self.feature_queue = Queue()
        self.output_queue = Queue(maxsize=self.batch_size * 2)

        self.frame_batch = []
        self.frame_count = len(self.frame_list_cycle)
        self.frame_index = 0
        self.thread_pool = thread_pool
        self.event_loop = event_loop
        self.monitor_thread = None
        self.monitor_running = False
        self.debug = False

    def mirror_index(self, index):
        turn = index // self.frame_count
        res = index % self.frame_count
        if turn % 2 == 0:
            return res
        else:
            return self.frame_count - res - 1

    def update_index(self, n):
        self.frame_index += n

    def warmup(self):
        for _ in range(self.config.warmup_iters):
            frame = self.get_audio_frame()
            self.frame_queue.put(frame)
            self.frame_batch.append(frame.get("data"))
        self._print_queue_status("Warmup completed")

    def put_audio_data(self, data: Data):
        self.input_queue.put(data, timeout=1)

    def get_audio_frame(self) -> Data:
        try:
            data = self.input_queue.get(timeout=0.01)
            audio_frame = data.get('data')
            state = 1
        except queue.Empty:
            audio_frame = np.zeros(self.chunk_size, dtype=np.float32)
            state = 0
        return Data(
            data=audio_frame,
            state=state,
        )

    def process_audio_frame_worker(self):
        while not self.stop_event.is_set():
            try:
                for _ in range(self.batch_size * 2):
                    frame = self.get_audio_frame()
                    self.frame_queue.put(frame, timeout=0.1)  # 添加超时
                    self.frame_batch.append(frame.get("data"))
                    # time.sleep(self.frame_interval)
                audio_feature_batch = self.model.encode_audio_feature(self.frame_batch, self.config)
                self.frame_batch = self.frame_batch[self.batch_size * 2:]
                self.feature_queue.put(
                    Data(data=audio_feature_batch),
                    timeout=0.1
                )

                # # 每处理一批数据后打印队列状态
                # self._print_queue_status("After processing audio frame batch")
            except Exception as e:
                print(f"Audio processing error: {e}")
                traceback.print_exc()
                time.sleep(0.1)

    def infer_video_frame_worker(self):
        count = 0
        counttime = 0
        while not self.stop_event.is_set():
            try:
                try:
                    audio_feature_data = self.feature_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                audio_feature_batch = audio_feature_data.get("data")
                audio_frames = []
                for _ in range(self.batch_size * 2):
                    frame_data = self.frame_queue.get(timeout=0.1)
                    audio_frames.append(frame_data)

                silence = all(frame.get("state") == 0 for frame in audio_frames)
                t = time.perf_counter()


                if silence:
                    for i in range(self.batch_size):
                        video_frame = None
                        audio_frame = audio_frames[i * 2:i * 2 + 2]
                        frame_index = self.mirror_index(self.frame_index)
                        self.output_queue.put(
                            (video_frame, audio_frame, frame_index),
                            timeout=0.1
                        )
                        self.update_index(1)
                else:
                    face_img_batch = []
                    for i in range(self.batch_size):
                        frame_index = self.mirror_index(self.frame_index + i)
                        face_img = self.face_list_cycle[frame_index]
                        face_img_batch.append(face_img)

                    # 模型推理
                    face_img_batch = np.asarray(face_img_batch)
                    audio_feature_batch = np.asarray(audio_feature_batch)
                    face_img_batch = torch.FloatTensor(face_img_batch).to(DEFAULT_RUNTIME_CONFIG.device)
                    audio_feature_batch = torch.FloatTensor(audio_feature_batch).to(DEFAULT_RUNTIME_CONFIG.device)

                    try:
                        with torch.no_grad():
                            pred_img_batch = self.model.inference(audio_feature_batch, face_img_batch, self.config)
                    except Exception as e:
                        print(f"Inference error: {e}")
                        traceback.print_exc()
                        continue
                    # 处理推理结果
                    for i, video_frame in enumerate(pred_img_batch):
                        audio_frame = audio_frames[i * 2:i * 2 + 2]
                        frame_index = self.mirror_index(self.frame_index)
                        self.output_queue.put(
                            (video_frame, audio_frame, frame_index),
                            timeout=0.1
                        )
                        self.update_index(1)
                count += self.batch_size
                counttime += (time.perf_counter() - t)
                # _totalframe += 1
                if count >= 100:
                    print(f"------actual avg infer fps:{count / counttime:.4f}")
                    count = 0
                    counttime = 0

                    # 每批推理完成后打印队列状态
                    self._print_queue_status("After inference batch")
            except Exception as e:
                print(f"Video inference error: {e}")
                traceback.print_exc()
                time.sleep(0.1)  # 出错后短暂休眠

    def process_output_frames_worker(self):
        while not self.stop_event.is_set():
            try:
                # 非阻塞获取输出帧
                try:
                    output_frame = self.output_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                video_frame, audio_frame, frame_index = output_frame

                # 处理视频帧
                if audio_frame[0].get('state') == 0 and audio_frame[1].get('state') == 0:
                    new_frame = self.frame_list_cycle[frame_index]
                else:
                    new_frame = copy.deepcopy(self.frame_list_cycle[frame_index])
                    bbox = self.coord_list_cycle[frame_index]
                    y1, y2, x1, x2 = bbox

                    if video_frame is not None:
                        video_frame = cv2.resize(video_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                        new_frame[y1:y2, x1:x2] = video_frame

                # 创建视频帧
                image = new_frame
                image[0, :] &= 0xFE  # 确保第一行是偶数，避免某些视频问题
                new_frame = VideoFrame.from_ndarray(image, format="bgr24")
                # asyncio.run_coroutine_threadsafe(self.video_track.put_frame(new_frame), self.event_loop)
                self.v_res_queue.put(new_frame)

                # 处理音频帧
                for frame_data in audio_frame:
                    frame, state = frame_data.get('data'), frame_data.get('state')
                    frame = (frame * 32767).astype(np.int16)
                    new_audio_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                    new_audio_frame.planes[0].update(frame.tobytes())
                    new_audio_frame.sample_rate = self.sample_rate
                    # print(new_audio_frame)
                    # asyncio.run_coroutine_threadsafe(self.audio_track.put_frame(new_frame), self.event_loop)
                    self.a_res_queue.put(new_frame)

                # # 每处理一帧后打印队列状态
                # self._print_queue_status("After processing output frame")
            except Exception as e:
                print(f"Output processing error: {e}")
                traceback.print_exc()
                time.sleep(0.1)  # 出错后短暂休眠

    def start(self):
        self.warmup()

        # 启动队列监控线程
        if self.debug:
            self.monitor_running = True
            self.monitor_thread = threading.Thread(
                target=self._queue_monitor,
                daemon=True
            )
            self.monitor_thread.start()

        self.thread_pool.submit(
            self.process_audio_frame_worker,
            task_info=TaskInfo(name="player.process_audio_frame_worker")
        ),
        self.thread_pool.submit(
            self.infer_video_frame_worker,
            task_info=TaskInfo(name="player.infer_video_frame_worker")
        ),
        self.thread_pool.submit(
            self.process_output_frames_worker,
            task_info=TaskInfo(name="player.process_output_frames_worker")
        )
        self._print_queue_status("Player started")

    def shutdown(self):
        # 设置停止事件
        self.stop_event.set()
        self.monitor_running = False  # 停止监控线程

        # 清空队列，避免阻塞
        queues = [self.input_queue, self.frame_queue, self.feature_queue, self.output_queue]
        for q in queues:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    continue
        self._print_queue_status("Player stopped, queues cleared")

    def _print_queue_status(self, message: str = "Queue status"):
        """打印当前所有队列的大小"""
        input_size = self.input_queue.qsize()
        frame_size = self.frame_queue.qsize()
        feature_size = self.feature_queue.qsize()
        output_size = self.output_queue.qsize()

        print(f"[{time.strftime('%H:%M:%S')}] {message}:")
        print(f"  Input queue size: {input_size}")
        print(f"  Frame queue size: {frame_size}")
        print(f"  Feature queue size: {feature_size}")
        print(f"  Output queue size: {output_size}")
        print()

    def _queue_monitor(self):
        """定期监控队列大小的后台线程"""
        while self.monitor_running:
            self._print_queue_status("Periodic queue monitor")
            time.sleep(5)  # 每5秒监控一次


import sys
import time
import queue
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QSlider, QVBoxLayout, QHBoxLayout, QWidget, QStatusBar,
                             QGroupBox, QGridLayout)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import av
import pyaudio
import traceback


class VideoThread(QThread):
    """视频处理线程，负责从队列中获取视频帧并发送显示信号"""
    frame_ready = pyqtSignal(np.ndarray)
    fps_updated = pyqtSignal(float)

    def __init__(self, video_frame_queue: queue.Queue):
        super().__init__()
        self.video_frame_queue = video_frame_queue
        self.running = False
        self.paused = False
        self.fps = 0.0
        self.last_time = time.time()
        self.frame_count = 0

    def run(self):
        """线程运行函数"""
        self.running = True
        self.last_time = time.time()
        self.frame_count = 0

        while self.running:
            if not self.paused:
                try:
                    # 从队列中获取视频帧
                    frame = self.video_frame_queue.get(timeout=0.01)
                    if isinstance(frame, av.VideoFrame):
                        # 将VideoFrame转换为numpy数组
                        img = frame.to_ndarray(format='bgr24')
                        self.frame_ready.emit(img)

                        # 计算FPS
                        current_time = time.time()
                        self.frame_count += 1
                        if current_time - self.last_time >= 1.0:
                            self.fps = self.frame_count / (current_time - self.last_time)
                            self.fps_updated.emit(self.fps)
                            self.frame_count = 0
                            self.last_time = current_time
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"视频线程错误: {e}")
                    print(traceback.format_exc())
            else:
                # 暂停状态，减少CPU占用
                time.sleep(0.1)

    def pause(self):
        """暂停视频播放"""
        self.paused = True

    def resume(self):
        """恢复视频播放"""
        self.paused = False

    def stop(self):
        """停止线程运行"""
        self.running = False
        self.wait()


class AudioThread(QThread):
    """音频处理线程，使用PyAudio播放音频帧"""
    volume_updated = pyqtSignal(float)
    audio_error = pyqtSignal(str)

    def __init__(self, audio_frame_queue: queue.Queue):
        super().__init__()
        self.audio_frame_queue = audio_frame_queue
        self.running = False
        self.paused = False
        self.volume = 0.5
        self.sample_rate = 44100
        self.channels = 1
        self.format = pyaudio.paInt16
        self.chunk_size = 1024  # 每次播放的样本数
        self.audio_buffer = queue.Queue(maxsize=50)  # 音频缓冲区
        self.pyaudio = None
        self.stream = None
        self.running_flag = False

    def run(self):
        """线程运行函数"""
        try:
            # 初始化PyAudio
            self.pyaudio = pyaudio.PyAudio()
            self.stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size
            )
            self.running_flag = True
            print("PyAudio 初始化成功")

            self.running = True
            while self.running:
                if not self.paused:
                    self._play_audio()
                else:
                    time.sleep(0.1)  # 暂停时减少CPU占用

        except Exception as e:
            self.audio_error.emit(f"音频线程错误: {e}")
            print(traceback.format_exc())
        finally:
            self._cleanup()

    def _play_audio(self):
        """播放音频缓冲区中的数据"""
        try:
            # 从缓冲区获取音频数据
            if self.audio_buffer.empty():
                # 发送静音数据避免卡顿
                silence = np.zeros(self.chunk_size, dtype=np.int16).tobytes()
                self.stream.write(silence, self.chunk_size)
                return

            pcm_data = self.audio_buffer.get(timeout=0.05)
            self.audio_buffer.task_done()

            # 应用音量
            volume_data = (np.frombuffer(pcm_data, dtype=np.int16) * self.volume).astype(np.int16)
            self.stream.write(volume_data.tobytes(), len(volume_data) // 2)  # 除以2因为是int16

        except queue.Empty:
            pass
        except Exception as e:
            self.audio_error.emit(f"播放音频错误: {e}")

    def _process_audio_frame(self, frame: av.AudioFrame):
        """处理音频帧并放入缓冲区"""
        try:
            # 确保音频格式为s16
            if frame.format.name != 's16':
                frame = frame.reformat(format='s16')

            # 转换为numpy数组
            pcm_data = frame.to_ndarray()

            # 处理多声道为单声道
            if len(pcm_data.shape) > 1:
                pcm_data = pcm_data.mean(axis=1)  # 简单混合为单声道

            # 转换为int16并应用音量
            pcm_data = (pcm_data * self.volume * 32767).astype(np.int16)

            # 放入缓冲区
            self.audio_buffer.put(pcm_data.tobytes(), block=False)

        except Exception as e:
            self.audio_error.emit(f"处理音频帧错误: {e}")

    def set_volume(self, volume: float):
        """设置音量 (0.0-1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        self.volume_updated.emit(self.volume)
        print(f"音量设置: {self.volume * 100:.0f}%")

    def pause(self):
        """暂停音频播放"""
        self.paused = True
        if self.stream:
            self.stream.stop_stream()
            print("音频已暂停")

    def resume(self):
        """恢复音频播放"""
        self.paused = False
        if self.stream and not self.stream.is_active():
            self.stream.start_stream()
            print("音频已恢复")

    def _cleanup(self):
        """清理资源"""
        if self.running_flag:
            self.running = False
            self.running_flag = False

            if self.stream:
                self.stream.stop_stream()
                self.stream.close()

            if self.pyaudio:
                self.pyaudio.terminate()
                print("PyAudio 资源已释放")

    def stop(self):
        """停止音频线程运行"""
        self._cleanup()
        self.wait()
        print("音频线程已停止")


class VideoAudioPlayer(QMainWindow):
    """支持音视频播放的主窗口"""

    def __init__(self, video_frame_queue: queue.Queue, audio_frame_queue: queue.Queue):
        super().__init__()
        self.video_frame_queue = video_frame_queue
        self.audio_frame_queue = audio_frame_queue
        self.video_thread = VideoThread(video_frame_queue)
        self.audio_thread = AudioThread(audio_frame_queue)
        self.playing = False

        self.init_ui()

        # 连接信号和槽
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.fps_updated.connect(self.update_fps)
        self.audio_thread.volume_updated.connect(self.update_volume_display)
        self.audio_thread.audio_error.connect(self.display_audio_error)

        self.play_button.clicked.connect(self.toggle_play)
        self.volume_slider.sliderMoved.connect(self.set_volume)
        self.slider.sliderMoved.connect(self.seek)

        # 启动音视频线程
        self.video_thread.start()
        self.audio_thread.start()

    def init_ui(self):
        """初始化用户界面"""
        # 设置窗口标题和大小
        self.setWindowTitle("AV Video & Audio Player (PyAudio)")
        self.setGeometry(100, 100, 800, 700)

        # 创建中央部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 创建视频显示标签
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setText("等待视频帧...")
        main_layout.addWidget(self.video_label)

        # 创建控制布局
        control_layout = QHBoxLayout()

        # 创建播放按钮
        self.play_button = QPushButton("Play")
        control_layout.addWidget(self.play_button)

        # 创建进度条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(0)
        control_layout.addWidget(self.slider)

        # 添加控制布局到主布局
        main_layout.addLayout(control_layout)

        # 创建音频控制组
        audio_control_group = QGroupBox("音频控制")
        audio_layout = QGridLayout()

        # 创建音量滑块
        self.volume_label = QLabel("音量: 50%")
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)

        audio_layout.addWidget(self.volume_label, 0, 0)
        audio_layout.addWidget(self.volume_slider, 0, 1)

        audio_control_group.setLayout(audio_layout)
        main_layout.addWidget(audio_control_group)

        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.fps_label = QLabel("FPS: 0.0")
        self.status_bar.addWidget(self.fps_label)
        self.volume_status = QLabel("音量: 50%")
        self.status_bar.addPermanentWidget(self.volume_status)
        self.error_label = QLabel("")
        self.status_bar.addPermanentWidget(self.error_label)

    def update_frame(self, frame: np.ndarray):
        """更新视频帧显示"""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def update_fps(self, fps: float):
        """更新FPS显示"""
        self.fps_label.setText(f"FPS: {fps:.2f}")

    def update_volume_display(self, volume: float):
        """更新音量显示"""
        volume_percent = int(volume * 100)
        self.volume_label.setText(f"音量: {volume_percent}%")
        self.volume_status.setText(f"音量: {volume_percent}%")

    def set_volume(self, value: int):
        """设置音量"""
        volume = value / 100.0
        self.audio_thread.set_volume(volume)

    def toggle_play(self):
        """切换播放/暂停状态"""
        if self.playing:
            self.video_thread.pause()
            self.audio_thread.pause()
            self.play_button.setText("Play")
        else:
            self.video_thread.resume()
            self.audio_thread.resume()
            self.play_button.setText("Pause")
        self.playing = not self.playing

    def seek(self, value):
        """视频定位功能（占位）"""
        print(f"定位到: {value}")

    def display_audio_error(self, error_msg: str):
        """显示音频错误信息"""
        self.error_label.setText(error_msg)
        print(f"音频错误: {error_msg}")

    def closeEvent(self, event):
        """窗口关闭事件处理"""
        # 停止音视频线程
        self.video_thread.stop()
        self.audio_thread.stop()
        event.accept()


if __name__ == '__main__':

    # 初始化pygame mixer
    app = QApplication(sys.argv)

    # 创建音视频帧队列
    video_frame_queue = queue.Queue()
    audio_frame_queue = queue.Queue()

    # 创建播放器窗口
    p = VideoAudioPlayer(video_frame_queue, audio_frame_queue)
    p.show()

    from engine.config import WAV2LIP_PLAYER_CONFIG
    from engine.human.avatar.wav2lip import Wav2LipWrapper, load_avatar
    from engine.runtime import thread_pool
    import time

    f = '../../../avatars/wav2lip256_avatar1'
    s_f = '../../../tests/test_datas/asr.wav'
    c_f = '../../../checkpoints/wav2lip.pth'
    model = Wav2LipWrapper(c_f)

    # 创建Player实例并启动
    player = Player(WAV2LIP_PLAYER_CONFIG, model, load_avatar(f), thread_pool, asyncio.get_event_loop(), video_frame_queue, audio_frame_queue)
    player.start()

    # 设置音频数据生产者
    def consume_fn(data):
        player.put_audio_data(data)
        return data

    from engine.human.voice.asr import soundfile_producer
    from engine.utils.pipeline import Pipeline

    pipeline = Pipeline(
        producer=soundfile_producer(s_f, fps=player.fps),
        consumer=consume_fn,
    )

    producer_task = thread_pool.submit(
        pipeline.produce_worker,
        task_info=TaskInfo(name="Task producer")
    )
    consumer_task = thread_pool.submit(
        pipeline.consume_worker,
        task_info=TaskInfo(name="Task consumer")
    )

    sys.exit(app.exec_())

    while True:
        if not producer_task.done() or not consumer_task.done():
            time.sleep(1)
            continue
        else:
            break
    # pipeline.shutdown()
    # player.shutdown()
    thread_pool.shutdown()

