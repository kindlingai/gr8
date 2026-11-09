#!/usr/bin/env python3
"""Browser E2E against only synthetic HTTP/TCP fixtures; saves desktop/mobile proof."""
import argparse
from pathlib import Path
import socket
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import uvicorn
from playwright.sync_api import sync_playwright, expect

from tools.console.server import weights
from tools.console.server.main import create_app
from tools.console.server.service import Console
from tools.console.tests.fakes import Fixture


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    fixture = Fixture()
    console = Console(fixture.console_config, fixture)
    app = create_app(console=console)
    sock = socket.socket()
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    started = threading.Event()
    original_startup = server.startup

    async def startup(*args, **kwargs):
        await original_startup(*args, **kwargs)
        started.set()
    server.startup = startup
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    if not started.wait(10):
        raise RuntimeError("fixture API did not start")
    errors = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1366, "height": 768}, device_scale_factor=1)
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            page.goto(f"http://localhost:{port}")
            expect(page.locator(".name")).to_have_text("GLM-5.3")
            expect(page.get_by_role("row").filter(has_text="Decode").locator(".v")).not_to_contain_text("—", timeout=10000)
            expect(page.get_by_text("SYNTHETIC FIXTURE", exact=False).first).to_be_visible()
            expect(page.locator(".pill.preset")).to_have_text("custom", timeout=10000)
            expect(page.locator(".pb.on")).to_have_count(0)
            expect(page.get_by_role("button", name="Knobs")).to_have_count(0)
            expect(page.get_by_text("same 3.25 bpw size, bits placed differently")).to_be_visible()
            for width in (1366, 390):
                page.set_viewport_size({"width": width, "height": 844 if width == 390 else 768})
                page.screenshot(path=str(output / f"page-{width}.png"), full_page=True)
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"horizontal overflow: {width}"
            print("PASS desktop 1366 + mobile 390: one page, three presets, no horizontal overflow")

            page.set_viewport_size({"width": 390, "height": 844})
            page.get_by_role("button", name="Speed").click()
            expect(page.get_by_role("dialog")).to_be_visible()
            expect(page.get_by_role("dialog")).to_contain_text("SESS=4")
            expect(page.get_by_role("dialog")).to_contain_text("SESS removed")
            expect(page.get_by_role("dialog")).to_contain_text("launcher defaults")
            page.get_by_role("button", name="Cancel", exact=True).click()
            assert fixture.calls == []
            page.get_by_role("button", name="Quality").click()
            expect(page.get_by_role("dialog")).to_contain_text("VARIANT=bit325")
            expect(page.get_by_role("dialog")).to_contain_text("SESS removed")
            page.keyboard.press("Escape")
            expect(page.get_by_role("dialog")).not_to_be_visible()
            assert fixture.calls == []
            print("PASS preset cancel/Escape: leftover SESS is listed and nothing is written")

            saved = {box: fixture.texts["glm53"][box] for box in fixture.texts["glm53"]}
            for box in fixture.texts["glm53"]:
                fixture.texts["glm53"][box] += "CAP=1000\n"

            def presets_body(response):
                return "/api/presets/glm53" in response.url and response.status == 200

            with page.expect_response(lambda response: presets_body(response) and "CAP=1000" in response.text(), timeout=10000):
                pass
            page.get_by_role("button", name="Quality").click()
            expect(page.get_by_role("dialog")).to_contain_text("CAP=1000")
            expect(page.get_by_role("dialog")).to_contain_text("CAP removed")
            page.get_by_role("button", name="Cancel", exact=True).click()
            for box, text in saved.items():
                fixture.texts["glm53"][box] = text
            with page.expect_response(lambda response: presets_body(response) and "CAP=1000" not in response.text(), timeout=10000):
                pass
            expect(page.locator(".pill.preset")).to_have_text("custom")
            assert fixture.calls == []
            print("PASS explicit CAP is custom and the confirm dialog shows its removal")

            page.get_by_role("button", name="Quality").click()
            expect(page.get_by_role("dialog")).to_contain_text("SESS=4")
            expect(page.get_by_role("dialog")).to_contain_text("SESS removed")
            page.screenshot(path=str(output / "review-390.png"), full_page=True)
            page.get_by_role("button", name="Restart now", exact=True).click()
            expect(page.get_by_label("Operation status")).to_contain_text("succeeded", timeout=15000)
            expect(page.locator(".pb.on")).to_contain_text("Quality", timeout=15000)
            assert [x[0] for x in fixture.calls] == ["write"] * 4 + ["stop", "switch"]
            assert all("VARIANT=bit325" in t and "KEEP_ME=1" in t and "SESS=" not in t for t in fixture.texts["glm53"].values())
            page.get_by_role("button", name="Quality").click()
            expect(page.get_by_role("dialog")).to_have_count(0)
            print("PASS confirmed preset: writes all four envs, drops SESS, stops, calls switch, verifies ready identity")

            fixture.missing_weights = {"variant": "home", "size_gb": 29.7, "rate_mb_s": 28}
            sentence = weights.download_text(29.7, 28, 900 + weights.extra_seconds(29.7, 28))
            with page.expect_response(lambda response: presets_body(response) and sentence in response.text(), timeout=10000):
                pass
            page.get_by_role("button", name="Speed").click()
            expect(page.get_by_role("dialog")).to_contain_text(sentence)
            expect(page.get_by_role("button", name="Restart now", exact=True)).to_be_enabled()
            page.screenshot(path=str(output / "download-390.png"), full_page=True)
            page.get_by_role("button", name="Restart now", exact=True).click()
            expect(page.get_by_label("Operation status")).to_contain_text("succeeded", timeout=15000)
            expect(page.get_by_label("Operation status")).to_contain_text("downloading")
            expect(page.get_by_label("Operation status")).to_contain_text(sentence)
            assert fixture.last_health_timeout == 900 + weights.extra_seconds(29.7, 28)
            assert fixture.last_health_timeout > 900
            page.get_by_role("button", name="Restore previous settings", exact=True).click()
            page.get_by_role("button", name="Restore settings now", exact=True).click()
            expect(page.get_by_label("Operation status")).to_contain_text("restore · glm53")
            expect(page.get_by_label("Operation status")).to_contain_text("succeeded", timeout=15000)
            assert all("VARIANT=bit325" in text for text in fixture.texts["glm53"].values())
            fixture.missing_weights = None
            with page.expect_response(lambda response: presets_body(response) and "downloads up to about" not in response.text(), timeout=10000):
                pass
            print("PASS missing variant weights ask for the download and extend the health wait")

            row = page.get_by_role("row").filter(has_text="DeepSeek-V4.1-Flash")
            row.get_by_role("button", name="Switch", exact=True).click()
            page.get_by_role("button", name="Switch now", exact=True).click()
            expect(page.get_by_label("Operation status")).to_contain_text("switch · ds41")
            expect(page.get_by_label("Operation status")).to_contain_text("succeeded", timeout=15000)
            assert fixture.active == "ds41"
            row.get_by_role("button", name="Stop", exact=True).click()
            page.get_by_role("button", name="Stop now", exact=True).click()
            expect(page.get_by_label("Operation status")).to_contain_text("stop · ds41")
            expect(page.get_by_label("Operation status")).to_contain_text("succeeded", timeout=15000)
            assert fixture.active is None
            print("PASS UI model switch and confirmed stop")

            fixture.fail_boot = True
            row = page.get_by_role("row").filter(has_text="GLM-5.3").filter(has_not_text="Flash")
            row.get_by_role("button", name="Switch", exact=True).click()
            page.get_by_role("button", name="Switch now", exact=True).click()
            expect(page.get_by_label("Operation status")).to_contain_text("failed", timeout=15000)
            expect(page.get_by_label("Operation status")).to_contain_text("readiness timeout")
            page.screenshot(path=str(output / "failure-390.png"), full_page=True)
            fixture.fail_boot = False
            print("PASS failed boot remains failed, not success")

            row.get_by_role("button", name="Switch", exact=True).click()
            page.get_by_role("button", name="Switch now", exact=True).click()
            expect(page.get_by_label("Operation status")).to_contain_text("succeeded", timeout=15000)
            expect(page.locator(".pb.on")).to_contain_text("Quality", timeout=15000)
            page.get_by_role("button", name="Speed").click()
            expect(page.get_by_role("dialog")).to_contain_text("VARIANT=bit325")
            expect(page.get_by_role("dialog")).to_contain_text("launcher defaults")
            fixture.fail_boot = True
            page.get_by_role("button", name="Restart now", exact=True).click()
            expect(page.get_by_label("Operation status")).to_contain_text("failed", timeout=15000)
            page.get_by_role("button", name="Restore previous settings", exact=True).click()
            page.get_by_role("button", name="Restore settings now", exact=True).click()
            expect(page.get_by_label("Operation status")).to_contain_text("restore · glm53")
            expect(page.get_by_label("Operation status")).to_contain_text("succeeded", timeout=15000)
            assert fixture.active is None
            assert all("VARIANT=bit325" in text for text in fixture.texts["glm53"].values())
            page.screenshot(path=str(output / "restored-settings-390.png"), full_page=True)
            fixture.fail_boot = False
            print("PASS failed preset restart can restore exact prior settings from UI without booting")

            page.route("**/api/snapshot", lambda route: route.abort())
            expect(page.get_by_text("Connection lost.", exact=False).first).to_be_visible(timeout=15000)
            assert all(button.is_disabled() for button in page.get_by_role("button", name="Switch", exact=True).all())
            page.screenshot(path=str(output / "disconnected-390.png"), full_page=True)
            page.unroute("**/api/snapshot")
            expect(page.get_by_text("Connection lost.", exact=False)).not_to_be_visible(timeout=15000)
            print("PASS disconnected stale warning and disabled actions; recovers after connection returns")
            assert not errors, errors
            print("PASS no browser JavaScript errors")
            browser.close()
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        fixture.close()
    print(f"Screenshots: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
