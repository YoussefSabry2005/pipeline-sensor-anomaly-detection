"""Capture a screenshot of the running dashboard with headless Chrome (optional helper).

    python scripts/capture_dashboard.py [url] [output.png]

Requires Google Chrome and a running dashboard. Uses the Chrome DevTools
protocol so the capture waits for the page to finish rendering.
"""

import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from tornado.websocket import websocket_connect

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9333


async def capture(url: str, out: Path, width: int, height: int, wait_s: float) -> None:
    for attempt in range(40):                      # wait for Chrome's debugging port
        try:
            targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json").read())
            break
        except OSError:
            if attempt == 39:
                raise
            await asyncio.sleep(0.5)
    ws = await websocket_connect(next(t for t in targets if t["type"] == "page")["webSocketDebuggerUrl"], max_message_size=2**28)
    counter = 0

    async def call(method: str, **params):
        nonlocal counter
        counter += 1
        await ws.write_message(json.dumps({"id": counter, "method": method, "params": params}))
        while True:
            msg = json.loads(await ws.read_message())
            if msg.get("id") == counter:
                return msg.get("result", {})

    await call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1.5, mobile=False)
    await call("Page.navigate", url=url)
    await asyncio.sleep(wait_s)
    shot = await call("Page.captureScreenshot", format="png")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(shot["data"]))
    print(f"Saved {out} ({out.stat().st_size / 1e3:.0f} kB)")


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "figures/dashboard/dashboard_overview.png")
    height = int(sys.argv[3]) if len(sys.argv) > 3 else 1250
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        chrome = subprocess.Popen(
            [CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
             "--hide-scrollbars", "--remote-allow-origins=*", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            asyncio.run(capture(url, out, 1500, height, wait_s=9))
        finally:
            chrome.terminate()
            chrome.wait(timeout=10)


if __name__ == "__main__":
    main()
