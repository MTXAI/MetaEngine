from engine.utils.concurrent import SharedFlag


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
    state = SharedFlag(StateReady)

    def __init__(self, state):
        self.state = SharedFlag(state)

    def swap_state(self, expected_state, new_state):
        return self.state.cas(expected_state, new_state)

    def set_state(self, state: int):
        self.state.set(state)

    def get_state(self):
        return self.state.get()
