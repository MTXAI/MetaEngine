import asyncio
import logging
import traceback

from aiohttp import web, WSMessage
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCRtpSender
from langchain_openai import ChatOpenAI

from engine.agent.agents.custom import KnowledgeAgent
from engine.agent.vecdb.chroma import try_load_db
from engine.config import WAV2LIP_PLAYER_CONFIG, DEFAULT_PROJECT_CONFIG, ONE_API_LLM_MODEL, \
    DEFAULT_VOICE_PROCESSOR_CONFIG, DEFAULT_AVATAR_PROCESSOR_CONFIG
from engine.human.avatar import wav2lip
from engine.human.avatar.avatar import AvatarProcessor
from engine.human.player import HumanPlayer
from engine.human.voice import AliTTSWrapper, EdgeTTSWrapper
from engine.human.voice.voice import VoiceProcessor
from engine.utils import Data

a_f = '../avatars/wav2lip256_avatar1'
c_f = '../checkpoints/wav2lip/wav2lip.pth'

tts_model_ali = AliTTSWrapper(
    model_str="cosyvoice-v1",
    api_key="sk-361f246a74c9421085d1d137038d5064",
    voice_type="longxiaochun",
    sample_rate=WAV2LIP_PLAYER_CONFIG.sample_rate,
)

tts_model_edge = EdgeTTSWrapper(
    voice_type="zh-CN-YunxiaNeural",
    sample_rate=WAV2LIP_PLAYER_CONFIG.sample_rate,
)

tts_models = [tts_model_ali, tts_model_edge]
tts_model_idx = 0

avatar = wav2lip.load_avatar(a_f)
avatar_model = wav2lip.Wav2LipWrapper(c_f, avatar)

# 创建Player实例并启动
loop = asyncio.new_event_loop()

# llm_model = ChatOpenAI(
#     model=QWEN_LLM_MODEL.model_id,
#     api_key=QWEN_LLM_MODEL.api_key,
#     base_url=QWEN_LLM_MODEL.api_base_url,
# )
llm_model = ChatOpenAI(
    model=ONE_API_LLM_MODEL.model_id,
    api_key=ONE_API_LLM_MODEL.api_key,
    base_url=ONE_API_LLM_MODEL.api_base_url,
)
# agent = SimpleAgent(llm_model)

vector_store = try_load_db(DEFAULT_PROJECT_CONFIG.vecdb_path, DEFAULT_PROJECT_CONFIG.docs_path)
agent = KnowledgeAgent(llm_model, vector_store)

voice_processor = VoiceProcessor(DEFAULT_VOICE_PROCESSOR_CONFIG)
avatar_processor = AvatarProcessor(DEFAULT_AVATAR_PROCESSOR_CONFIG)

player = HumanPlayer(
    config=WAV2LIP_PLAYER_CONFIG,
    agent=agent,
    tts_model=tts_models[tts_model_idx],
    avatar=avatar,
    avatar_model=avatar_model,
    voice_processor=voice_processor,
    avatar_processor=avatar_processor,
    loop=loop,
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
                data = msg.json()
                if data["type"] == "offer":
                    # 处理客户端的 offer
                    offer = RTCSessionDescription(sdp=data["sdp"], type="offer")
                    pc = RTCPeerConnection()

                    # 创建并添加音视频轨道
                    audio_track = player.get_audio_track()
                    video_track = player.get_video_track()

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

async def pause(request):
    res_data = player.pause()
    return web.json_response({"status": "success", "data": res_data})


# echo 接口
async def echo(request):
    data = await request.json()
    text = data.get('text')
    if text and not player.is_busy():
        res_data = player.put_text_data(
            Data(
                data=text,
                is_chat=False,
                stream=False,
            )
        )
        logging.info(res_data)
        return web.json_response({"status": "success", "data": res_data})

    return web.json_response({"status": "error", "message": "Missing text parameter"}, status=400)


# chat 接口
async def chat(request):
    data = await request.json()
    question = data.get('question')
    if question and not player.is_busy():
        res_data = player.put_text_data(
            Data(
                data=question,
                is_chat=True,
                stream=True,
            )
        )
        logging.info(res_data)
        return web.json_response({"status": "success", "data": res_data})
    return web.json_response({"status": "error", "message": "Missing question parameter"}, status=400)


# 主函数
def main():
    """启动 Web 服务器"""
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_post("/pause", pause)
    app.router.add_post("/echo", echo)
    app.router.add_post("/chat", chat)

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
