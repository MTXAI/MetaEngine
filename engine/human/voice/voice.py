from torch import nn


class TTSModelWrapper(nn.Module):
    def flush(self):
        """
        清理本地文本和音频缓存
        :return:
        """
        pass

    def streaming_inference(self, text_block, stream=False):
        """
        接收 text streaming, 先在本地组装文本块, 再输出音频
        :param text_block: 流式输入的文本块
        :param stream: 是否流式输出音频
        :return:
        """
        pass

    def inference(self, text, stream=False):
        """
        接收 text, 输出音频
        :param text: 文本
        :param stream: 是否流式输出音频
        :return:
        """
        pass
