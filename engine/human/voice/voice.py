from torch import nn


class TTSModelWrapper(nn.Module):
    def streaming_inference(self, text):
        """
        流式输入文本, 输出音频
        :param text:
        :return:
        """
        raise NotImplementedError()

    def inference(self, text):
        """
        输入文本, 输出音频
        :param text:
        :return:
        """
        pass
