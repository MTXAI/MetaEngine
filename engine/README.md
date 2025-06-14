
human
- asr: 语音转文本, Automatic Speech Recognition
- svc: 语音变声, Speech Voice Conversion
- tts: 文本转语音, Text to Speech
- talk: 语音转视频
- agent: 回答问题/生成文本
- aug: 文本/语音和视频数据的增强与处理
- - 文本, 分词/安全词等
- - 语音, 随机噪声/语气增强等
- - 视频, 随机噪声/背景增强等

数据转换
- 语音 - asr - 文本
- 文本 - tts - 语音
- 语音 - svc - 语音
- 语音 - talk - 视频
- 文本 - agent - 文本

流水线
- 文字对话: agent - tts - svc - talk - player
- 语音对话: asr - agent - tts - svc - talk - player
- 语音朗读: tts - svc - talk - player

human_player: 按照流水线 得到语音和视频, 然后合成视频


protocol
- web_rtc
- srs
- rtmp
- rtc_push

task: 构建任务流水线, 基于 human 和 third_party 工具, 实现具体任务, 然后通过 protocol 处理

输入和输出
- 输入: 文本/语音
- 输出: 视频/视频流
