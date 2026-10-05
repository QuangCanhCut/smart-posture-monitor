import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "app" / "templates" / "index.html"
STATIC_ROOT = PROJECT_ROOT / "app" / "static"


def _index_html() -> str:
    return INDEX_PATH.read_text(encoding="utf-8")


def test_frontend_runtime_assets_are_local() -> None:
    html = _index_html()
    resource_urls = re.findall(
        r"(?:src|href)=[\"']([^\"']+)[\"']|@import url\([\"']([^\"']+)[\"']\)",
        html,
    )
    urls = [left or right for left, right in resource_urls]

    assert not any(url.startswith(("https://", "http://")) for url in urls)
    for url in urls:
        if url.startswith("/static/"):
            assert (STATIC_ROOT / url.removeprefix("/static/")).is_file()


def test_frame_sender_is_response_driven_without_interval_queue() -> None:
    html = _index_html()

    assert "let frameInFlight = false" in html
    assert "if (isFrameResult) sendFrame();" in html
    assert "frameInFlight = true" in html
    assert not re.search(r"setInterval\s*\(\s*sendFrame", html)
    assert not re.search(r"requestAnimationFrame\s*\(\s*sendFrame", html)


def test_reconnect_and_render_loops_have_cleanup_guards() -> None:
    html = _index_html()

    assert "let reconnectTimer = null" in html
    assert "if (!isPageUnloading && !reconnectTimer)" in html
    assert 'window.addEventListener("beforeunload"' in html
    assert "cancelAnimationFrame(renderFrameRequestId)" in html


def test_dashboard_work_is_bounded() -> None:
    html = _index_html()

    assert "statsRenderNow - lastStatsRenderAt >= 1000" in html
    assert "renderNow - lastVideoRenderAt >= 1000 / 30" in html
    assert "postureChart.data.labels.length > 120" in html
    assert "postureChart.data.labels.shift()" in html

