import asyncio
import traceback

from aiohttp import web, WSMessage
import json
import logging
import ssl

from aiohttp.web_ws import WebSocketResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCRtpSender

from engine.config import WAV2LIP_PLAYER_CONFIG
from engine.human.player.player import HumanPlayer
from engine.human.player.track import VideoStreamTrack, StreamTrackSync
from engine.config import WAV2LIP_PLAYER_CONFIG
from engine.human.avatar.wav2lip import Wav2LipWrapper, load_avatar
from engine.runtime import thread_pool
from engine.utils.pool import TaskInfo
from engine.human.voice.asr import soundfile_producer

f = '../avatars/wav2lip256_avatar1'
s_f = '../tests/test_datas/asr_example.wav'
c_f = '../checkpoints/wav2lip.pth'

model = Wav2LipWrapper(c_f)

# 创建Player实例并启动
loop = asyncio.new_event_loop()

player = HumanPlayer(
    config=WAV2LIP_PLAYER_CONFIG,
    model=model,
    avatar=load_avatar(f),
    loop=loop,
    audio_producer=soundfile_producer(s_f, fps=10)
)

# 存储已连接的客户端
connected_clients = set()


# 静态文件处理
async def index(request):
    """返回前端 HTML 页面"""
    with open("./test_web/index.html", "r") as f:
        content = f.read()
    return web.Response(content_type="text/html", text=content)


# WebRTC 信令处理
async def websocket_handler(request):
    """处理 WebSocket 连接，用于 WebRTC 信令交换"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)


    # 创建新的 RTCPeerConnection
    pc = RTCPeerConnection()
    client_id = id(pc)
    connected_clients.add(client_id)
    logging.info(f"客户端 {client_id} 已连接")


    # 处理 ICE 候选
    @pc.on("icecandidate")
    async def on_icecandidate(candidate):
        if candidate:
            await ws.send_json({
                "type": "candidate",
                "candidate": candidate.to_dict()
            })

    # 处理连接状态变化
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logging.info(f"连接状态: {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            connected_clients.discard(client_id)
            await pc.close()

    # 接收客户端消息
    try:
        async for msg in ws:
            msg: WSMessage
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)

                if data["type"] == "offer":
                    # 处理客户端的 offer
                    offer = RTCSessionDescription(sdp=data["sdp"], type="offer")
                    pc = RTCPeerConnection()

                    # 创建并添加音视频轨道
                    audio_track = player.audio_track
                    video_track = player.video_track

                    pc.addTrack(audio_track)
                    pc.addTrack(video_track)
                    # 确保轨道已正确初始化
                    if not audio_track.kind or not video_track.kind:
                        raise ValueError("媒体轨道未正确初始化")

                    # 明确设置轨道方向为sendonly（服务器只发送，客户端只接收）
                    audio_track.direction = "sendonly"
                    video_track.direction = "sendonly"

                    capabilities = RTCRtpSender.getCapabilities("video")
                    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
                    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
                    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
                    transceiver = pc.getTransceivers()[1]
                    transceiver.setCodecPreferences(preferences)

                    await pc.setRemoteDescription(offer)

                    # 创建并发送 answer
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)

                    await ws.send_json({
                        "type": "answer",
                        "sdp": pc.localDescription.sdp
                    })

                    logging.info(f"向客户端 {client_id} 发送 answer")

                # elif data["type"] == "candidate":
                #     # 处理客户端的 ICE 候选
                #     candidate = data["candidate"]
                #     if candidate:
                #         await pc.addIceCandidate(candidate)
                #         logging.info(f"从客户端 {client_id} 收到 ICE 候选")
            elif msg.type == web.WSMsgType.ERROR:
                logging.error(f"WebSocket 错误: {ws.exception()}")

    except Exception as e:
        logging.error(f"处理客户端 {client_id} 时发生错误: {str(e)}")
        traceback.print_exc()
    finally:
        # 清理资源
        connected_clients.discard(client_id)
        await pc.close()
        logging.info(f"客户端 {client_id} 已断开连接")

    return ws


# 主函数
def main():
    """启动 Web 服务器"""
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", websocket_handler)

    # 启动服务器
    runner = web.AppRunner(app)
    asyncio.set_event_loop(loop)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    loop.run_until_complete(site.start())

    player.start()

    logging.info("服务器已启动，访问 http://localhost:8080")
    loop.run_forever()



if __name__ == "__main__":
    logging.info("start...")
    main()
