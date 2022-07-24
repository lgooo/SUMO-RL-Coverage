import time


class Logger:
    def __init__(self, enabled=True):
        self.history = []
        self.initial_time = time.time()
        self.enabled = enabled

    def log(self, message):
        if not self.enabled:
            return
        self.history.append((message, time.time() - self.initial_time))

    def reset(self):
        if not self.enabled:
            return
        self.history = []
        self.initial_time = time.time()

    def digest(self):
        if not self.enabled:
            return []
        previous_time = 0
        ret = []
        for message, t in self.history:
            ret.append((message, t - previous_time))
            previous_time = t
        return ret
