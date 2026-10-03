"""
Temporal Smoothing + Yaw Gating cho Smart Posture Monitor.

Module độc lập, KHÔNG cần retrain model. Xử lý post-prediction:

1. Temporal Smoothing:
   - Majority vote trên cửa sổ trượt (configurable 5-10 giây).
   - Hysteresis: cần N frame liên tục đồng thuận mới chuyển trạng thái.
   - Triệt tiêu prediction nhảy cóc (flickering).

2. Yaw Gating (Phát hiện quay đầu bằng D4 = face_rotation_proxy):
   - Mode Conservative: khi head_turn detected → giữ nguyên trạng thái cũ.
   - Mode Informative: khi head_turn detected → trả về "head_turned".

Cách dùng:
    from src.temporal_monitor import TemporalMonitor

    monitor = TemporalMonitor(
        fps=5,
        window_seconds=5,
        hysteresis_frames=3,
        yaw_mode="conservative",
        yaw_threshold=0.35,
    )

    # Trong vòng lặp realtime:
    smoothed_label = monitor.update(
        raw_prediction="forward_slouch",
        face_rotation_proxy=0.42,
    )
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Literal, Optional


# ============================================================
# Constants
# ============================================================

VALID_LABELS = [
    "correct",
    "forward_slouch",
    "lean_left",
    "lean_right",
]

HEAD_TURNED_LABEL = "head_turned"

# Giá trị mặc định cho yaw threshold.
#
# face_rotation_proxy = dist(left_eye, nose) / dist(right_eye, nose)
#
# Khi nhìn thẳng camera (từ góc 45°):
#     ratio ≈ 0.8–1.2 (tùy vóc dáng, vị trí ngồi)
#
# Khi quay đầu sang phải (xa camera):
#     ratio tăng > 1.5 (mắt trái xa mũi hơn nhiều so với mắt phải)
#
# Khi quay đầu sang trái (gần camera):
#     ratio giảm < 0.5 (mắt trái gần mũi hơn)
#
# Ngưỡng mặc định 0.35:
#     Kiểm tra |log(ratio)| > log(threshold_high/threshold_low)
#     hoặc đơn giản hơn: ratio > HIGH hoặc ratio < LOW
#
# Hai ngưỡng mặc định:
DEFAULT_YAW_LOW = 0.45   # ratio < 0.45 → quay đầu sang trái
DEFAULT_YAW_HIGH = 1.80  # ratio > 1.80 → quay đầu sang phải


# ============================================================
# Dataclass cho state tracking
# ============================================================

@dataclass
class MonitorState:
    """
    Trạng thái nội bộ của TemporalMonitor tại mỗi thời điểm.
    Có thể đọc để debug/logging.
    """

    current_label: str = "evaluating"
    raw_prediction: Optional[str] = None
    smoothed_prediction: str = "evaluating"
    is_head_turned: bool = False
    yaw_value: Optional[float] = None
    yaw_gated: bool = False
    frame_count: int = 0
    hysteresis_counter: int = 0
    pending_label: Optional[str] = None


# ============================================================
# TemporalMonitor
# ============================================================

class TemporalMonitor:
    """
    Bộ xử lý hậu dự đoán (post-prediction processor).

    Kết hợp:
    - Temporal smoothing (majority vote + hysteresis)
    - Yaw gating (phát hiện quay đầu, chặn prediction sai)

    Hoàn toàn stateful: mỗi lần gọi update() là 1 frame mới.
    """

    def __init__(
        self,
        fps: int = 5,
        window_seconds: float = 5.0,
        window_size: Optional[int] = None,
        min_samples: int = 3,
        hysteresis_frames: int = 3,
        yaw_mode: Literal["conservative", "informative", "off"] = "conservative",
        yaw_low: float = DEFAULT_YAW_LOW,
        yaw_high: float = DEFAULT_YAW_HIGH,
    ):
        """
        Args:
            fps:
                Số frame per second của hệ thống inference.
                Dùng để tính kích thước cửa sổ trượt.

            window_seconds:
                Độ dài cửa sổ trượt (giây).
                Mặc định 5 giây → 25 frames @ 5fps.

            hysteresis_frames:
                Số frame liên tục mà prediction mới phải chiếm đa số
                trong cửa sổ trước khi chuyển trạng thái.
                Giá trị cao → ổn định hơn nhưng phản hồi chậm hơn.

            yaw_mode:
                "conservative" : khi quay đầu → giữ trạng thái cũ.
                "informative"  : khi quay đầu → trả về "head_turned".
                "off"          : tắt yaw gating.

            yaw_low:
                Ngưỡng dưới cho face_rotation_proxy.
                ratio < yaw_low → quay đầu sang trái.

            yaw_high:
                Ngưỡng trên cho face_rotation_proxy.
                ratio > yaw_high → quay đầu sang phải.
        """

        if fps <= 0:
            raise ValueError(f"fps phải > 0, nhận được {fps}")
        if window_seconds <= 0:
            raise ValueError(
                f"window_seconds phải > 0, nhận được {window_seconds}"
            )
        if hysteresis_frames < 0:
            raise ValueError(
                f"hysteresis_frames phải >= 0, nhận được {hysteresis_frames}"
            )
        if window_size is not None and window_size <= 0:
            raise ValueError(
                f"window_size phải > 0, nhận được {window_size}"
            )
        if min_samples <= 0:
            raise ValueError(f"min_samples phải > 0, nhận được {min_samples}")
        if yaw_low >= yaw_high:
            raise ValueError(
                f"yaw_low ({yaw_low}) phải < yaw_high ({yaw_high})"
            )

        self.fps = fps
        # ``window_size`` is intended for realtime inference whose throughput is
        # variable.  Omitting it preserves the legacy fps * seconds behaviour.
        self.window_size = (
            int(window_size)
            if window_size is not None
            else max(1, int(fps * window_seconds))
        )
        self.min_samples = min(min_samples, self.window_size)
        self.hysteresis_frames = hysteresis_frames
        self.yaw_mode = yaw_mode
        self.yaw_low = yaw_low
        self.yaw_high = yaw_high

        # Cửa sổ trượt chứa các prediction gần nhất.
        self._window: deque[str] = deque(maxlen=self.window_size)

        # Trạng thái hiện tại.
        self._state = MonitorState()

    @property
    def state(self) -> MonitorState:
        """Truy cập trạng thái hiện tại (read-only)."""
        return self._state

    def reset(self) -> None:
        """Reset toàn bộ trạng thái về ban đầu."""
        self._window.clear()
        self._state = MonitorState()

    def _detect_head_turn(
        self,
        face_rotation_proxy: Optional[float],
    ) -> bool:
        """
        Kiểm tra có đang quay đầu hay không dựa trên face_rotation_proxy.

        Returns:
            True nếu phát hiện quay đầu.
        """

        if self.yaw_mode == "off":
            return False

        if face_rotation_proxy is None:
            # Không có giá trị D4 → không thể phát hiện
            # → an toàn: coi như không quay đầu.
            return False

        return (
            face_rotation_proxy < self.yaw_low
            or face_rotation_proxy > self.yaw_high
        )

    def _majority_vote(self) -> str:
        """
        Tìm label xuất hiện nhiều nhất trong cửa sổ trượt.

        Nếu hòa, ưu tiên "correct" (an toàn cho người dùng:
        không spam cảnh báo khi không chắc chắn).
        """

        if not self._window:
            return "evaluating"

        counter = Counter(self._window)
        max_count = counter.most_common(1)[0][1]

        # Lấy tất cả labels có số lần xuất hiện bằng max.
        top_labels = [
            label for label, count in counter.items()
            if count == max_count
        ]

        # Ưu tiên "correct" khi hòa.
        if "correct" in top_labels:
            return "correct"

        # Nếu không có "correct", ưu tiên trạng thái hiện tại.
        if self._state.current_label in top_labels:
            return self._state.current_label

        # Fallback: label đầu tiên trong danh sách.
        return top_labels[0]

    def _apply_hysteresis(
        self,
        candidate_label: str,
    ) -> str:
        """
        Áp dụng hysteresis: chỉ chuyển trạng thái khi candidate_label
        giữ vững liên tục trong hysteresis_frames frame.

        Returns:
            Label sau khi áp dụng hysteresis.
        """

        current = self._state.current_label

        # Không cần hysteresis nếu candidate = current.
        if candidate_label == current:
            self._state.hysteresis_counter = 0
            self._state.pending_label = None
            return current

        # Candidate khác current → đếm.
        if self._state.pending_label == candidate_label:
            self._state.hysteresis_counter += 1
        else:
            # Candidate mới, reset counter.
            self._state.pending_label = candidate_label
            self._state.hysteresis_counter = 1

        # Đã đủ số frame liên tục → chuyển.
        if self._state.hysteresis_counter >= self.hysteresis_frames:
            self._state.hysteresis_counter = 0
            self._state.pending_label = None
            return candidate_label

        # Chưa đủ → giữ nguyên.
        return current

    def update(
        self,
        raw_prediction: str,
        face_rotation_proxy: Optional[float] = None,
    ) -> str:
        """
        Xử lý một frame mới.

        Args:
            raw_prediction:
                Kết quả dự đoán thô từ model SVM.
                Phải là một trong: correct, forward_slouch, lean_left, lean_right.

            face_rotation_proxy:
                Giá trị feature D4 (face_rotation_proxy) của frame hiện tại.
                Dùng cho yaw gating. None nếu không có (tắt yaw gating).

        Returns:
            Label cuối cùng sau khi áp dụng smoothing + gating.
            Có thể là "head_turned" nếu yaw_mode="informative".
        """

        self._state.frame_count += 1
        self._state.raw_prediction = raw_prediction
        self._state.yaw_value = face_rotation_proxy

        # Invalid/control states must never contaminate the vote window.  Keep
        # the last stable posture (or evaluating during startup).
        if raw_prediction not in VALID_LABELS:
            self._state.is_head_turned = False
            self._state.yaw_gated = False
            return self._state.current_label

        # --------------------------------------------------------
        # BƯỚC 1: Yaw Gating
        # --------------------------------------------------------
        is_head_turned = self._detect_head_turn(face_rotation_proxy)
        self._state.is_head_turned = is_head_turned

        if is_head_turned:
            self._state.yaw_gated = True

            if self.yaw_mode == "informative":
                # Không thêm prediction vào window (frame bị bỏ qua).
                return HEAD_TURNED_LABEL

            elif self.yaw_mode == "conservative":
                # Giữ trạng thái cũ, không thêm vào window.
                return self._state.current_label

        self._state.yaw_gated = False

        # --------------------------------------------------------
        # BƯỚC 2: Thêm prediction vào cửa sổ trượt
        # --------------------------------------------------------
        self._window.append(raw_prediction)

        # Do not claim the user is sitting correctly before enough valid model
        # predictions have been observed.
        if len(self._window) < self.min_samples:
            self._state.smoothed_prediction = "evaluating"
            return "evaluating"

        # --------------------------------------------------------
        # BƯỚC 3: Majority vote
        # --------------------------------------------------------
        candidate = self._majority_vote()
        self._state.smoothed_prediction = candidate

        # Bootstrap the first stable label directly once the minimum sample
        # count is reached. Hysteresis applies to subsequent transitions.
        if self._state.current_label == "evaluating":
            self._state.current_label = candidate
            self._state.hysteresis_counter = 0
            self._state.pending_label = None
            return candidate

        # --------------------------------------------------------
        # BƯỚC 4: Hysteresis
        # --------------------------------------------------------
        final_label = self._apply_hysteresis(candidate)
        self._state.current_label = final_label

        return final_label

    def get_stats(self) -> dict:
        """
        Trả về thống kê hiện tại để debug/logging.
        """

        window_list = list(self._window)
        counter = Counter(window_list) if window_list else {}
        majority_count = max(counter.values(), default=0)
        vote_confidence = (
            majority_count / len(window_list)
            if window_list and len(window_list) >= self.min_samples
            else None
        )
        display_count = counter.get(self._state.current_label, 0)
        display_confidence = (
            display_count / len(window_list)
            if (
                window_list
                and len(window_list) >= self.min_samples
                and self._state.current_label in VALID_LABELS
            )
            else None
        )

        return {
            "frame_count": self._state.frame_count,
            "current_label": self._state.current_label,
            "raw_prediction": self._state.raw_prediction,
            "smoothed_prediction": self._state.smoothed_prediction,
            "is_head_turned": self._state.is_head_turned,
            "yaw_gated": self._state.yaw_gated,
            "yaw_value": self._state.yaw_value,
            "window_size": len(self._window),
            "window_max_size": self.window_size,
            "window_distribution": dict(counter),
            "valid_samples": len(self._window),
            "min_samples": self.min_samples,
            "vote_confidence": vote_confidence,
            "display_confidence": display_confidence,
            "hysteresis_counter": self._state.hysteresis_counter,
            "pending_label": self._state.pending_label,
        }

    def __repr__(self) -> str:
        return (
            f"TemporalMonitor("
            f"fps={self.fps}, "
            f"window={self.window_size} frames, "
            f"hysteresis={self.hysteresis_frames}, "
            f"yaw_mode='{self.yaw_mode}', "
            f"state='{self._state.current_label}'"
            f")"
        )


# ============================================================
# CLI demo
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TemporalMonitor — Demo")
    print("=" * 60)

    monitor = TemporalMonitor(
        fps=5,
        window_seconds=5,
        hysteresis_frames=3,
        yaw_mode="conservative",
    )

    print(f"Config: {monitor}")
    print(f"Window size: {monitor.window_size} frames")
    print()

    # Mô phỏng chuỗi prediction.
    demo_sequence = [
        # 10 frame ngồi đúng.
        ("correct", 1.0),
        ("correct", 1.0),
        ("correct", 0.95),
        ("correct", 1.05),
        ("correct", 0.98),
        ("correct", 1.0),
        ("correct", 0.97),
        ("correct", 1.02),
        ("correct", 1.0),
        ("correct", 0.99),

        # 3 frame nhảy cóc (nhiễu) → smoothing phải chặn.
        ("lean_right", 1.1),
        ("forward_slouch", 0.9),
        ("correct", 1.0),

        # Thực sự cúi gù → cần 3+ frame liên tục.
        ("forward_slouch", 0.85),
        ("forward_slouch", 0.80),
        ("forward_slouch", 0.82),
        ("forward_slouch", 0.79),
        ("forward_slouch", 0.81),
        ("forward_slouch", 0.83),

        # Quay đầu → yaw gating.
        ("lean_right", 2.5),     # D4 = 2.5 → quay đầu phải
        ("lean_left", 0.3),      # D4 = 0.3 → quay đầu trái
        ("correct", 1.0),        # Quay lại bình thường.
    ]

    print("Frame | Raw Prediction   | D4    | Output           | Notes")
    print("------|------------------|-------|------------------|------")

    for i, (raw_pred, d4_value) in enumerate(demo_sequence, start=1):
        result = monitor.update(
            raw_prediction=raw_pred,
            face_rotation_proxy=d4_value,
        )

        stats = monitor.get_stats()
        notes = []

        if stats["yaw_gated"]:
            notes.append("YAW_GATED")
        if stats["pending_label"]:
            notes.append(f"pending={stats['pending_label']}")
        if stats["hysteresis_counter"] > 0:
            notes.append(f"hyst={stats['hysteresis_counter']}")

        note_str = ", ".join(notes) if notes else ""

        print(
            f"  {i:3d} | {raw_pred:16s} | {d4_value:5.2f} | "
            f"{result:16s} | {note_str}"
        )

    print()
    print("Final state:", monitor.get_stats())
