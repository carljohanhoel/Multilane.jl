class ParallelJuliaTest:
    def __init__(self, state):
        self.state = state

    def print_state(self):
        print(self.state)

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state