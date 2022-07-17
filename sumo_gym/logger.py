import time


class Logger:
    def __init__(self):
        self.history = []
        self.initial_time = time.time()

    def log(self, message):
        self.history.append((message, time.time() - self.initial_time))

    def reset(self):
        self.history = []
        self.initial_time = time.time()

    def digest(self):
        previous_time = 0
        ret = []
        for message, t in self.history:
            ret.append((message, t - previous_time))
            previous_time = t
        return ret
