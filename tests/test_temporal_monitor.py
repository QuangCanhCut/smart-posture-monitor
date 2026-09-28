"""
Unit tests cho src/temporal_monitor.py.

Kiểm tra:
- Temporal smoothing (majority vote + hysteresis)
- Yaw gating (conservative / informative / off)
- Edge cases (empty window, tie-breaking, reset)

Chạy:
    pytest tests/test_temporal_monitor.py -v
"""

import pytest

from src.temporal_monitor import (
    TemporalMonitor,
    HEAD_TURNED_LABEL,
    DEFAULT_YAW_LOW,
    DEFAULT_YAW_HIGH,
)


# ============================================================
# 1. FIXTURE
# ============================================================

@pytest.fixture
def monitor_fast() -> TemporalMonitor:
    """
    Monitor với window nhỏ (5 frames) và hysteresis thấp (2 frames)
    để test nhanh hơn.
    """
    return TemporalMonitor(
        fps=5,
        window_seconds=1.0,  # 5 frames
        hysteresis_frames=2,
        yaw_mode="conservative",
    )


@pytest.fixture
def monitor_no_hysteresis() -> TemporalMonitor:
    """Monitor không có hysteresis."""
    return TemporalMonitor(
        fps=5,
        window_seconds=1.0,
        hysteresis_frames=0,
        yaw_mode="off",
    )


@pytest.fixture
def monitor_informative() -> TemporalMonitor:
    """Monitor với yaw_mode = informative."""
    return TemporalMonitor(
        fps=5,
        window_seconds=1.0,
        hysteresis_frames=2,
        yaw_mode="informative",
    )


# ============================================================
# 2. TESTS: Khởi tạo
# ============================================================

class TestInit:

    def test_window_size_dung(self):
        m = TemporalMonitor(fps=5, window_seconds=5.0)
        assert m.window_size == 25

    def test_window_size_1_second(self):
        m = TemporalMonitor(fps=10, window_seconds=1.0)
        assert m.window_size == 10

    def test_fps_phai_duong(self):
        with pytest.raises(ValueError, match="fps"):
            TemporalMonitor(fps=0)

    def test_window_phai_duong(self):
        with pytest.raises(ValueError, match="window_seconds"):
            TemporalMonitor(fps=5, window_seconds=0)

    def test_hysteresis_phai_khong_am(self):
        with pytest.raises(ValueError, match="hysteresis_frames"):
            TemporalMonitor(fps=5, hysteresis_frames=-1)

    def test_yaw_low_phai_nho_hon_high(self):
        with pytest.raises(ValueError, match="yaw_low"):
            TemporalMonitor(fps=5, yaw_low=2.0, yaw_high=1.0)

    def test_default_state_la_correct(self):
        m = TemporalMonitor(fps=5)
        assert m.state.current_label == "correct"
        assert m.state.frame_count == 0


# ============================================================
# 3. TESTS: Smoothing cơ bản
# ============================================================

class TestSmoothing:

    def test_prediction_on_dinh_giu_nguyen(
        self, monitor_no_hysteresis: TemporalMonitor
    ):
        """Nếu tất cả prediction giống nhau → output giống."""
        m = monitor_no_hysteresis
        for _ in range(10):
            result = m.update("correct")
        assert result == "correct"

    def test_nhieu_1_frame_bi_chan(
        self, monitor_fast: TemporalMonitor
    ):
        """1 frame nhiễu giữa chuỗi correct → bị smoothing chặn."""
        m = monitor_fast

        # 4 frame correct.
        for _ in range(4):
            m.update("correct")

        # 1 frame nhiễu.
        result = m.update("lean_right")

        # Vẫn phải là correct (majority vote + hysteresis).
        assert result == "correct"

    def test_chuyen_trang_thai_khi_du_frame(
        self, monitor_no_hysteresis: TemporalMonitor
    ):
        """
        Khi majority chuyển sang label mới (không hysteresis)
        → output phải chuyển theo.
        """
        m = monitor_no_hysteresis

        # Window size = 5. Fill 3 forward_slouch + 2 correct.
        m.update("correct")
        m.update("correct")
        m.update("forward_slouch")
        m.update("forward_slouch")
        result = m.update("forward_slouch")

        # 3 FS vs 2 correct → FS thắng.
        assert result == "forward_slouch"


# ============================================================
# 4. TESTS: Hysteresis
# ============================================================

class TestHysteresis:

    def test_hysteresis_chan_chuyen_nhanh(
        self, monitor_fast: TemporalMonitor
    ):
        """
        Hysteresis = 2. Candidate mới phải giữ 2 frame liên tục
        mới được chuyển.
        """
        m = monitor_fast

        # Fill window toàn correct.
        for _ in range(5):
            m.update("correct")

        # Giờ gửi toàn forward_slouch.
        # Frame 1: candidate = correct (majority vẫn correct)
        r1 = m.update("forward_slouch")
        assert r1 == "correct"

        # Frame 2: candidate vẫn correct (3 correct, 2 fs)
        r2 = m.update("forward_slouch")
        assert r2 == "correct"

        # Frame 3: candidate = forward_slouch (2 correct, 3 fs)
        # Nhưng hysteresis counter mới = 1 → chưa chuyển.
        r3 = m.update("forward_slouch")
        assert r3 == "correct"

        # Frame 4: candidate vẫn = forward_slouch
        # Hysteresis counter = 2 → chuyển!
        r4 = m.update("forward_slouch")
        assert r4 == "forward_slouch"

    def test_hysteresis_reset_khi_doi_candidate(self):
        """
        Nếu candidate thay đổi giữa chừng → reset counter.
        """
        m = TemporalMonitor(
            fps=5,
            window_seconds=2.0,   # 10 frames
            hysteresis_frames=3,
            yaw_mode="off",
        )

        # Fill window toàn correct.
        for _ in range(10):
            m.update("correct")

        # Bắt đầu gửi forward_slouch.
        for _ in range(5):
            m.update("forward_slouch")

        # Giữa chừng chuyển sang lean_left.
        for _ in range(5):
            m.update("lean_left")

        # Kiểm tra counter đã reset: pending phải là lean_left.
        stats = m.get_stats()
        assert stats["pending_label"] in (None, "lean_left")


# ============================================================
# 5. TESTS: Yaw Gating
# ============================================================

class TestYawGating:

    def test_conservative_giu_trang_thai_cu(
        self, monitor_fast: TemporalMonitor
    ):
        """
        Yaw mode = conservative.
        Khi quay đầu → giữ trạng thái cũ (correct).
        """
        m = monitor_fast

        # Fill correct.
        for _ in range(5):
            m.update("correct", face_rotation_proxy=1.0)

        # Quay đầu phải (D4 = 2.5 > yaw_high).
        result = m.update("lean_right", face_rotation_proxy=2.5)
        assert result == "correct"
        assert m.state.is_head_turned is True
        assert m.state.yaw_gated is True

    def test_conservative_khong_them_vao_window(
        self, monitor_fast: TemporalMonitor
    ):
        """
        Khi yaw gated → prediction KHÔNG được thêm vào window.
        """
        m = monitor_fast

        for _ in range(5):
            m.update("correct", face_rotation_proxy=1.0)

        # Gửi 3 frame quay đầu.
        for _ in range(3):
            m.update("lean_right", face_rotation_proxy=2.5)

        stats = m.get_stats()
        # Window chỉ có 5 frame correct, không có lean_right.
        assert stats["window_distribution"].get("lean_right", 0) == 0

    def test_informative_tra_ve_head_turned(
        self, monitor_informative: TemporalMonitor
    ):
        """
        Yaw mode = informative.
        Khi quay đầu → trả về "head_turned".
        """
        m = monitor_informative

        for _ in range(5):
            m.update("correct", face_rotation_proxy=1.0)

        # Quay đầu trái (D4 = 0.3 < yaw_low).
        result = m.update("lean_left", face_rotation_proxy=0.3)
        assert result == HEAD_TURNED_LABEL

    def test_yaw_off_khong_gate(self):
        """
        Yaw mode = off.
        Không gate, prediction thường bình thường.
        """
        m = TemporalMonitor(
            fps=5,
            window_seconds=1.0,
            hysteresis_frames=0,
            yaw_mode="off",
        )

        for _ in range(5):
            m.update("correct", face_rotation_proxy=1.0)

        # D4 vượt ngưỡng nhưng mode off → không gate.
        result = m.update("lean_right", face_rotation_proxy=2.5)
        # Vẫn là correct (majority vote) chứ không phải vì gating.
        assert result == "correct"

    def test_yaw_none_khong_gate(
        self, monitor_fast: TemporalMonitor
    ):
        """
        face_rotation_proxy = None → không gate.
        """
        m = monitor_fast

        for _ in range(5):
            m.update("correct")

        result = m.update("lean_right", face_rotation_proxy=None)
        assert m.state.is_head_turned is False

    def test_nguong_yaw_chinh_xac(self):
        """
        Kiểm tra ngưỡng chính xác: chỉ > high hoặc < low mới gate.
        """
        m = TemporalMonitor(
            fps=5,
            window_seconds=1.0,
            hysteresis_frames=0,
            yaw_mode="conservative",
            yaw_low=0.5,
            yaw_high=1.5,
        )

        # Đúng tại ngưỡng → không gate.
        m.update("correct", face_rotation_proxy=0.5)
        assert m.state.is_head_turned is False

        m.update("correct", face_rotation_proxy=1.5)
        assert m.state.is_head_turned is False

        # Dưới low → gate.
        m.update("lean_left", face_rotation_proxy=0.49)
        assert m.state.is_head_turned is True

        # Trên high → gate.
        m.update("lean_right", face_rotation_proxy=1.51)
        assert m.state.is_head_turned is True


# ============================================================
# 6. TESTS: Tie-breaking
# ============================================================

class TestTieBreaking:

    def test_hoa_uu_tien_correct(self):
        """Khi hòa, ưu tiên correct."""
        m = TemporalMonitor(
            fps=2,
            window_seconds=1.0,  # 2 frames
            hysteresis_frames=0,
            yaw_mode="off",
        )

        m.update("correct")
        result = m.update("forward_slouch")
        # 1 correct vs 1 FS → hòa → ưu tiên correct.
        assert result == "correct"

    def test_hoa_uu_tien_current_state(self):
        """Khi hòa không có correct, ưu tiên trạng thái hiện tại."""
        m = TemporalMonitor(
            fps=2,
            window_seconds=1.0,  # 2 frames
            hysteresis_frames=0,
            yaw_mode="off",
        )

        # Chuyển sang forward_slouch trước.
        m._state.current_label = "forward_slouch"
        m.update("forward_slouch")
        result = m.update("lean_right")

        # 1 FS vs 1 LR → hòa → ưu tiên current (FS).
        assert result == "forward_slouch"


# ============================================================
# 7. TESTS: Reset
# ============================================================

class TestReset:

    def test_reset_xoa_het_trang_thai(
        self, monitor_fast: TemporalMonitor
    ):
        """Reset phải xóa hết trạng thái."""
        m = monitor_fast

        for _ in range(10):
            m.update("forward_slouch")

        m.reset()

        assert m.state.frame_count == 0
        assert m.state.current_label == "correct"
        assert m.state.hysteresis_counter == 0
        assert m.state.pending_label is None

        stats = m.get_stats()
        assert stats["window_size"] == 0


# ============================================================
# 8. TESTS: get_stats
# ============================================================

class TestGetStats:

    def test_stats_co_du_keys(self, monitor_fast: TemporalMonitor):
        m = monitor_fast
        m.update("correct")
        stats = m.get_stats()

        expected_keys = {
            "frame_count",
            "current_label",
            "raw_prediction",
            "smoothed_prediction",
            "is_head_turned",
            "yaw_gated",
            "yaw_value",
            "window_size",
            "window_max_size",
            "window_distribution",
            "hysteresis_counter",
            "pending_label",
        }

        assert set(stats.keys()) == expected_keys

    def test_frame_count_tang_dung(
        self, monitor_fast: TemporalMonitor
    ):
        m = monitor_fast

        for i in range(7):
            m.update("correct")

        assert m.state.frame_count == 7


# ============================================================
# 9. TESTS: Kịch bản thực tế
# ============================================================

class TestRealScenario:

    def test_nguoi_dung_cui_gu_roi_ngoi_thang_lai(self):
        """
        Kịch bản: Ngồi thẳng → cúi gù → ngồi thẳng lại.
        Monitor phải phản ánh đúng nhưng có độ trễ.
        """
        m = TemporalMonitor(
            fps=5,
            window_seconds=2.0,  # 10 frames
            hysteresis_frames=3,
            yaw_mode="off",
        )

        # Phase 1: Ngồi thẳng (10 frames).
        for _ in range(10):
            assert m.update("correct") == "correct"

        # Phase 2: Bắt đầu cúi gù (liên tục forward_slouch).
        results_phase2 = []
        for _ in range(10):
            results_phase2.append(m.update("forward_slouch"))

        # Phải có độ trễ trước khi chuyển sang forward_slouch.
        assert results_phase2[0] == "correct"  # Chưa chuyển ngay.
        assert results_phase2[-1] == "forward_slouch"  # Cuối cùng phải chuyển.

        # Phase 3: Ngồi thẳng lại.
        results_phase3 = []
        for _ in range(10):
            results_phase3.append(m.update("correct"))

        assert results_phase3[-1] == "correct"

    def test_quay_dau_khong_tao_canh_bao_sai(self):
        """
        Kịch bản: Ngồi thẳng → quay đầu nhìn ngang → ngồi thẳng.
        Model SVM sẽ predict sai khi quay đầu.
        Yaw gating phải chặn prediction sai.
        """
        m = TemporalMonitor(
            fps=5,
            window_seconds=2.0,  # 10 frames
            hysteresis_frames=3,
            yaw_mode="conservative",
        )

        # Ngồi thẳng.
        for _ in range(10):
            m.update("correct", face_rotation_proxy=1.0)

        # Quay đầu → model predict sai là lean_right.
        for _ in range(5):
            result = m.update(
                "lean_right",
                face_rotation_proxy=2.5,
            )
            # Yaw gating phải giữ "correct".
            assert result == "correct"

        # Quay lại bình thường.
        result = m.update("correct", face_rotation_proxy=1.0)
        assert result == "correct"
