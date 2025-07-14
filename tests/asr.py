from funasr import AutoModel
from typing import List, Dict
import json

def format_recognition_result(res: List[Dict]) -> str:

    formatted_output = []

    for result in res:
        sentences = result["sentence_info"]

        formatted_output.append("语音识别结果：\n")
        for sentence in sentences:
            speaker_id = sentence["spk"]
            text = sentence["text"]
            start_time = sentence["start"] / 1000
            end_time = sentence["end"] / 1000

            formatted_sentence = (
                f"说话人 {speaker_id} "
                f"[{start_time:.2f}s - {end_time:.2f}s]: "
                f"{text}"
            )
            formatted_output.append(formatted_sentence)

        return "\n".join(formatted_output)


wav_file = "./test_datas/asr_example.wav"


## 离线ASR+说话人识别
model = AutoModel(
    model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", spk_model="cam++"
)

res = model.generate(
    input=wav_file,
    batch_size_s=300,
    hotword="魔搭",
)

print(format_recognition_result(res))


# 实时ASR+说话人识别(基于录音设备的识别)
model = AutoModel(model="paraformer-zh-streaming", hub='ms', disable_update=True)
# chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
