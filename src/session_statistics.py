import time

class SessionStatistics:

    def __init__(self):
        self.start_time = time.time()
        self.good_time = 0
        self.bad_time = 0
        self.warnings_count = 0

    def update(self, posture, warning=False):
        current_time = time.time()
        elapsed = current_time - self.start_time
        if posture == "GOOD POSTURE":
            self.good_time = elapsed
        else:
            self.bad_time = elapsed
        if warning:
            self.warnings_count += 1

    def get_statistics(self):
        return {
            "good_time": self.good_time,
            "bad_time": self.bad_time,
            "warnings_count": self.warnings_count
        }