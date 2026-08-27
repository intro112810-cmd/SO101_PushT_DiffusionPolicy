#!/usr/bin/env python3
"""Local web dashboard for PushT SO-100 collection progress.

Serves a browser page at http://127.0.0.1:PORT that auto-refreshes every
2 seconds and shows episodes/frames/last-save/progress for the dataset
being recorded by `collect-native --launch`.

Usage:
  python3 -B scripts/collection_web_dashboard.py --dataset-root <path> [--target 200] [--port 8888]
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from collection_dashboard import read_status  # same-dir read-only status logic

PAGE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>PushT SO-100 수집 대시보드</title>
<style>
  body { font-family: system-ui, sans-serif; background: #0f1420; color: #e8e8e8; margin: 0; padding: 24px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .muted { color: #8ba0c0; font-size: 13px; }
  .card { background: #1a2233; border: 1px solid #2c3a55; border-radius: 10px; padding: 20px 24px; margin: 16px 0; max-width: 720px; }
  .row { display: flex; justify-content: space-between; margin: 8px 0; }
  .big { font-size: 30px; font-weight: 700; }
  .bar { background: #2c3a55; border-radius: 6px; height: 14px; margin: 8px 0; overflow: hidden; }
  .bar > div { background: #4cc38a; height: 100%; transition: width 0.5s; }
  .alert { background: #3a2c1a; border: 1px solid #c38a4c; border-radius: 8px; padding: 10px 14px; margin: 8px 0; color: #ffd27a; font-weight: 600; }
</style>
</head>
<body>
  <h1>PushT SO-100 수집 대시보드</h1>
  <div class="muted" id="dataset">연결 중...</div>
  <div class="card">
    <div class="row"><span>에피소드</span><span class="big" id="episodes">-</span></div>
    <div class="row"><span>목표</span><span id="target">-</span></div>
    <div class="bar"><div id="bar" style="width:0%"></div></div>
    <div class="row"><span>진행률</span><span id="progress">-</span></div>
    <div class="row"><span>남은 에피소드</span><span id="remaining">-</span></div>
    <div class="row"><span>프레임</span><span id="frames">-</span></div>
    <div class="row"><span>마지막 저장</span><span id="last_saved">-</span></div>
  </div>
  <div id="events"></div>
  <script>
    let last = -1;
    async function refresh() {
      try {
        const r = await fetch('/api/status');
        const s = await r.json();
        if (s.ok) {
          document.getElementById('dataset').textContent = s.dataset;
          document.getElementById('episodes').textContent = s.episodes + ' / ' + s.target;
          document.getElementById('target').textContent = s.target;
          document.getElementById('bar').style.width = Math.min(100, s.progress) + '%';
          document.getElementById('progress').textContent = s.progress + '%';
          document.getElementById('remaining').textContent = s.remaining;
          document.getElementById('frames').textContent = s.frames + (s.fps ? ' (fps=' + s.fps + ')' : '');
          document.getElementById('last_saved').textContent = s.last_saved
            ? (s.last_saved + ' (' + s.seconds_since_save + 's 전)')
            : '(아직 없음)';
          if (last >= 0 && s.episodes > last) {
            const div = document.createElement('div');
            div.className = 'alert';
            div.textContent = '>>> 새 에피소드 저장됨: ' + last + ' -> ' + s.episodes + ' (+' + (s.episodes - last) + ')';
            document.getElementById('events').prepend(div);
          }
          last = s.episodes;
        } else {
          document.getElementById('dataset').textContent = s.error;
        }
      } catch (e) {
        document.getElementById('dataset').textContent = '대시보드 서버 연결 실패: ' + e;
      }
    }
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


class ServerState:
    dataset_root: Path
    target: int


SERVER = ServerState()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/status":
            status = read_status(SERVER.dataset_root, SERVER.target)
            body = json.dumps(status, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("[web-dashboard] %s\n" % (fmt % args))


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--target", type=int, default=200)
    p.add_argument("--port", type=int, default=8888)
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    SERVER.dataset_root = args.dataset_root
    SERVER.target = args.target
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[web-dashboard] listening on http://127.0.0.1:{args.port} (dataset={args.dataset_root})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[web-dashboard] stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
