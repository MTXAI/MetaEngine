from engine import runtime
from engine.config import PlayerConfig, HumanConfig
from engine.human.avatar import Avatar
from engine.human.character import Character
from engine.human.player import HumanPlayer
from engine.human.voice import Voice
from engine.utils import Data


class Human:
    def __init__(
            self, player_config, voice, avatar, character, transports
    ):
        self.player = HumanPlayer(
            config=player_config,
            character=character,
            avatar=avatar,
            voice=voice,
            loop=runtime.main_loop,
            transports=transports,
        )

    def change_voice(self, new_voice) -> bool:
        return self.player.set_voice(new_voice)

    def change_character(self, new_character) -> bool:
        return self.player.set_character(new_character)

    def startup(self):
        self.player.run()

    def pause(self):
        return self.player.pause()

    def shutdown(self):
        self.player.shutdown()

    def status(self) -> int:
        return self.player.state()

    def say(self, text, force=False) -> bool:
        data = Data(
            data=text,
            is_chat=False,
            stream=False,
        )
        res = self.player.speak(data, force=force)
        return res.ok

    def answer(self, question, force=False) -> bool:
        data = Data(
            data=question,
            is_chat=True,
            stream=True,
        )
        res = self.player.speak(data, force=force)
        return res.ok

    def execute(self):
        """
        todo 执行某些规划好的指令
        :return:
        """
        pass

