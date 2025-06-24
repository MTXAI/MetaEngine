
## engine
AI数字人框架

### agent
基于rag的LLM agent, 并支持知识库和短期记忆

### character
基于agent和human构建, 带有特定人设与外形的角色

### human
数字人各组成部分的定义, 包括数字人声音, 形象和合成

1. avatar: 数字人形象, 包括嘴形, 肢体动作, 眼神, 形象生成与复制等
2. player: 数字人合成与增强, 包括声音和形象视频的合成, 背景替换, 语音视频增强等
3. voice: 数字人声音, 包括文字转语音, 语音转文字, 语音变声, 语音复制等

### protocol
各类通信和传输协议工具实现

### service
基于数字人(agent+human+character)与 protocol, 包装为指定服务, 如推流服务或 http api 服务

### utils
各类工具代码

## models
各类AI模型的入口, 按照模型类型分类, 如llm, asr, svc, tts, talker等类别, 相同类型的模型尽量提供统一的接口. 还需具备一定的扩展能力


## tasks
基于engine和third_party实现具体的业务逻辑, 解决指定任务


## third_party
各类第三方平台或工具集成, 如直播平台的数据接口

