

# todo voice, avatar, character 都用 config 来构建, 包括 human 都有对应的 factory
class Human:
    """
    数字人构建、组件管理(不提供访问接口)、运行控制、状态查询与功能调用
    1. 视频功能调用, char+voice+avatar,如 pause, speak
    2. 语音功能调用, char+voice
    3. 文字功能调用, char
    """
    def __init__(self, voice=None, avatar=None, character=None):
        pass

    # ----------------- 构建与组件更新接口 -----------------
    def change_voice(self, new_voice):
        """
        更换数字人语音组件。
        :param new_voice: 新的语音组件实例
        """
        self.voice = new_voice

    def change_avatar(self, new_avatar):
        """
        更换数字人形象组件。
        :param new_avatar: 新的形象组件实例
        """
        self.avatar = new_avatar

    def change_character(self, new_character):
        """
        更换数字人人设组件。
        :param new_character: 新的人设组件实例
        """
        self.character = new_character

    # ----------------- 运行控制接口 -----------------
    def startup(self):
        pass

    def pause(self):
        pass

    def sleep(self):
        pass

    def shutdown(self):
        pass

    def reboot(self):
        pass

    # ----------------- 查询接口 -----------------
    def status(self):
        pass

    # ----------------- 功能调用接口 -----------------
    def say(self, text):
        pass

    def answer(self, question):
        pass

    def execute(self):
        pass

