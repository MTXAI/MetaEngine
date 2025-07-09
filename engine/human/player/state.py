from engine.utils.concurrent.lock import RWLock


StateNotReady = -1
StateReady = 0
StateBusy = 1
StateSpeaking = 2
StatePause = 3
state_str = {
    StateNotReady: "StateNotReady",
    StateReady: "StateReady",
    StateBusy: "StateBusy",
    StateSpeaking: "StateSpeaking",
    StatePause: "StatePause",
}


class HumanState:
    state: int = StateReady
    state_lock = RWLock()

    def __init__(self, state):
        self.state = state

    def swap_state(self, expected_state, new_state):
        if self.get_state() == expected_state:
            self.set_state(new_state)
            return True
        return False

    def set_state(self, state: int):
        with self.state_lock.writer_lock():
            self.state = state

    def get_state(self):
        with self.state_lock.reader_lock():
            return self.state
