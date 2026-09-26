import time


class Pomodoro_timer:

    def __init__(self):
        self.running = False
        self.paused = False

        self.start_time = None
        self.elapsed_before_pause = 0.0

    def start(self):
        """Bắt đầu một session mới."""
        # Nếu đang chạy thì không start lại
        if self.running:
            return

        self.start_time = time.time()
        self.elapsed_before_pause = 0.0

        self.running = True
        self.paused = False

    def pause(self):
        """Tạm dừng session."""
        if self.running and not self.paused:
            self.elapsed_before_pause += time.time() - self.start_time
            self.paused = True

    def resume(self):
        """Tiếp tục session."""
        if self.running and self.paused:
            self.start_time = time.time()
            self.paused = False

    def stop(self):
        """Kết thúc session."""
        if self.running and not self.paused:
            self.elapsed_before_pause += time.time() - self.start_time

        self.running = False
        self.paused = False

    def reset(self):
        """Reset timer về 0."""
        self.running = False
        self.paused = False
        self.start_time = None
        self.elapsed_before_pause = 0.0

    def get_elapsed_time(self):
        """Trả về tổng số giây của session."""

        if not self.running:
            return self.elapsed_before_pause

        if self.paused:
            return self.elapsed_before_pause

        return self.elapsed_before_pause + (
            time.time() - self.start_time
        )

    @staticmethod
    def format_time(seconds):
        """Đổi số giây thành HH:MM:SS."""

        seconds = int(seconds)

        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"