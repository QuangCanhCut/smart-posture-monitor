import time


class TemporalMonitor:

    def __init__(self, alert_time = 5):
        # Số giây cần ngồi sai liên tục trước khi cảnh báo
        self.alert_time = alert_time
        # Thời điểm bắt đầu phát hiện tư thế BAD
        self.bad_start_time = None
        # Trạng thái cảnh báo hiện tại
        self.warning = False

    def update(self, posture):
        """
        Nhận kết quả tư thế hiện tại:
        GOOD hoặc BAD.
        Trả về:
        warning: có cần cảnh báo hay không
        """

        # Nếu tư thế hiện tại là BAD
        if posture == "BAD POSTURE":

            # Nếu đây là lần đầu phát hiện BAD
            if self.bad_start_time is None:
                self.bad_start_time = time.time()
            # Tính thời gian đã ngồi sai liên tục
            bad_duration = time.time() - self.bad_start_time
            # Nếu vượt quá thời gian cảnh báo
            if bad_duration >= self.alert_time:
                self.warning = True

        else:
            # Nếu tư thế trở lại GOOD
            # thì reset bộ đếm thời gian
            self.bad_start_time = None
            self.warning = False

        return self.warning