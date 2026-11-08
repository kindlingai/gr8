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
            for width in (1366, 390):
                page.set_viewport_size({"width": width, "height": 844 if width == 390 else 768})
                for tab in ("Overview", "Knobs", "Models"):
                    page.get_by_role("button", name=tab, exact=True).click()
                    if tab == "Knobs":
                        expect(page.get_by_label("REUSE", exact=True)).to_be_visible()
                        if width == 390:
                            table = page.locator("table.knobs").bounding_box()
                            tag = page.locator("table.knobs tr.fastrow td").nth(2).bounding_box()
                            assert abs(tag["x"] + tag["width"] - table["x"] - table["width"]) < 2, "hidden description column still consumes mobile width"
                    page.screenshot(path=str(output / f"{tab.lower()}-{width}.png"), full_page=True)
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"horizontal overflow: {tab}/{width}"
            print("PASS desktop 1366 + mobile 390: three tabs, screenshots, no horizontal overflow")

            page.get_by_role("button", name="Knobs", exact=True).click()
            page.get_by_label("REUSE", exact=True).select_option("1")
            expect(page.get_by_text("1 change staged · needs restart")).to_be_visible()
            assert fixture.calls == [], "staging wrote a file"
            page.get_by_role("button", name="Discard", exact=True).click()
            expect(page.get_by_label("REUSE", exact=True)).to_have_value("")
            assert fixture.calls == []
            page.get_by_label("REUSE", exact=True).select_option("1")
            page.get_by_role("button", name="Review + restart…").click()
            expect(page.get_by_role("dialog")).to_be_visible()
            page.screenshot(path=str(output / "review-390.png"), full_page=True)
            page.get_by_role("button", name="Cancel", exact=True).click()
            assert fixture.calls == []
            page.get_by_role("button", name="Review + restart…").click()
            page.keyboard.press("Escape")
            expect(page.get_by_role("dialog")).not_to_be_visible()
            assert fixture.calls == []
            print("PASS stage/discard/cancel/Escape: no backend write or restart")

            page.get_by_role("button", name="Review + restart…").click()
            page.get_by_role("button", name="Restart now", exact=True).click()
            expect(page.get_by_label("Operation status")).to_contain_text("succeeded", timeout=15000)
            expect(page.get_by_text("1 change staged · needs restart")).not_to_be_visible(timeout=15000)
            assert [x[0] for x in fixture.calls] == ["write"] * 4 + ["stop", "switch"]
            assert all("REUSE=1" in t for t in fixture.texts["glm53"].values())
            print("PASS confirmed knob edit: writes all four envs, stops, calls switch, verifies ready identity")

            page.get_by_role("button", name="Models", exact=True).click()
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
            page.get_by_role("button", name="Knobs", exact=True).click()
            entered, release = threading.Event(), threading.Event()
            original_read = fixture.read_all
            def delayed_read(model):
                entered.set()
                release.wait(5)
                return original_read(model)
            fixture.read_all = delayed_read
            page.get_by_role("button", name="Refresh overrides", exact=True).click()
            assert entered.wait(3)
            try:
                expect(page.get_by_label("REUSE", exact=True)).to_be_disabled()
            finally:
                release.set()
                fixture.read_all = original_read
            expect(page.get_by_role("button", name="Refresh overrides", exact=True)).to_be_enabled()
            expect(page.get_by_label("REUSE", exact=True)).to_have_value("1")
            page.get_by_label("REUSE", exact=True).select_option("0")
            page.get_by_role("button", name="Review + restart…").click()
            fixture.fail_boot = True
            page.get_by_role("button", name="Restart now", exact=True).click()
            expect(page.get_by_label("Operation status")).to_contain_text("failed", timeout=15000)
            page.get_by_role("button", name="Restore previous settings", exact=True).click()
            page.get_by_role("button", name="Restore settings now", exact=True).click()
            expect(page.get_by_label("Operation status")).to_contain_text("restore · glm53")
            expect(page.get_by_label("Operation status")).to_contain_text("succeeded", timeout=15000)
            assert fixture.active is None
            assert all("REUSE=1" in text for text in fixture.texts["glm53"].values())
            page.screenshot(path=str(output / "restored-settings-390.png"), full_page=True)
            fixture.fail_boot = False
            page.get_by_role("button", name="Models", exact=True).click()
            print("PASS failed knob restart can restore exact prior settings from UI without booting")

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
