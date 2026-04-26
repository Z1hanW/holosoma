#!/usr/bin/env python3
"""Small browser control panel to start/reset a split MuJoCo tracking rollout."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from aiohttp import ClientSession, WSMsgType, web
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[3]
INFER_SRC_ROOT = REPO_ROOT / "src" / "holosoma_inference"
for path in (REPO_ROOT / "src" / "holosoma", INFER_SRC_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from holosoma_inference.utils.sim_control import PolicyControlPush, SimControlPush  # noqa: E402


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MuJoCo Track Trigger</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101216;
      color: #f3f4f6;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at top, rgba(42, 54, 76, 0.9), transparent 55%),
        linear-gradient(180deg, #141922, #0f1218);
    }
    main {
      width: min(920px, calc(100vw - 32px));
      display: grid;
      gap: 16px;
      padding: 24px 0;
    }
    h1 {
      margin: 0;
      font-size: 32px;
      font-weight: 700;
    }
    .panel {
      border: 1px solid #2b3342;
      border-radius: 14px;
      background: rgba(20, 25, 34, 0.92);
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.35);
      overflow: hidden;
    }
    .hero {
      display: grid;
      gap: 18px;
      padding: 28px;
      text-align: center;
    }
    .hint {
      color: #b6becd;
      font-size: 15px;
    }
    .actions {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 280px));
      gap: 14px;
      justify-content: center;
    }
    .trigger {
      width: min(280px, 100%);
      justify-self: center;
      height: 150px;
      border: 1px solid #34506f;
      border-radius: 20px;
      background:
        linear-gradient(180deg, rgba(52, 93, 138, 0.9), rgba(26, 50, 79, 0.98));
      color: #f8fafc;
      cursor: pointer;
      font: inherit;
      transition: transform 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
      box-shadow: 0 14px 34px rgba(17, 28, 46, 0.42);
    }
    .trigger.reset {
      border-color: #714437;
      background:
        linear-gradient(180deg, rgba(134, 73, 49, 0.9), rgba(75, 41, 32, 0.98));
    }
    .trigger:hover {
      border-color: #7db2f0;
      transform: translateY(-1px);
    }
    .trigger.reset:hover {
      border-color: #f1a477;
    }
    .trigger:active,
    .trigger.active {
      transform: translateY(1px) scale(0.99);
      box-shadow: 0 8px 18px rgba(17, 28, 46, 0.3);
    }
    .trigger strong {
      display: block;
      font-size: 54px;
      line-height: 1;
      margin-bottom: 10px;
    }
    .trigger span {
      display: block;
      font-size: 18px;
      letter-spacing: 0.02em;
    }
    .status {
      display: grid;
      gap: 10px;
      padding: 0 28px 28px;
      color: #d7dce6;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 14px;
      overflow-wrap: anywhere;
    }
    .scene {
      display: none;
      min-height: 420px;
      background: #0b0d12;
    }
    .scene.active {
      display: block;
    }
    iframe {
      width: 100%;
      height: min(64vh, 760px);
      min-height: 420px;
      border: 0;
      display: block;
      background: #0b0d12;
    }
    a {
      color: #8fc3ff;
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
    @media (max-width: 640px) {
      .actions {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
<main>
  <section class="panel hero">
    <h1>MuJoCo Track Trigger</h1>
    <div class="hint">
      Press <strong>__TRACK_KEY_LABEL__</strong> to start tracking,
      or <strong>__RESET_KEY_LABEL__</strong> / <strong>Backspace</strong> to reset the rollout.
    </div>
    <div class="actions">
      <button id="triggerButton" class="trigger" type="button">
        <strong>__TRACK_KEY_LABEL__</strong>
        <span>Start Track</span>
      </button>
      <button id="resetButton" class="trigger reset" type="button">
        <strong>__RESET_KEY_LABEL__</strong>
        <span>Reset Rollout</span>
      </button>
    </div>
  </section>
  <section class="panel">
    <div class="status">
      <div id="policyStatus">policy: waiting for __TRACK_KEY_LABEL__</div>
      <div id="message">Press __TRACK_KEY_LABEL__ to start motion + policy.</div>
      <div id="sceneLink"></div>
    </div>
  </section>
  <section id="scenePanel" class="panel scene">
    <iframe id="sceneFrame" title="MuJoCo scene"></iframe>
  </section>
</main>
<script>
const trackKey = __TRACK_KEY_JSON__;
const resetKey = __RESET_KEY_JSON__;
const trackAction = __TRACK_ACTION_JSON__;
const appBaseUrl = new URL("./", window.location.href);
const triggerButton = document.getElementById("triggerButton");
const resetButton = document.getElementById("resetButton");
const policyStatus = document.getElementById("policyStatus");
const message = document.getElementById("message");
const sceneUrlRaw = __SCENE_URL_JSON__;
const sceneUrl = sceneUrlRaw ? new URL(sceneUrlRaw, appBaseUrl).toString() : "";
const scenePanel = document.getElementById("scenePanel");
const sceneFrame = document.getElementById("sceneFrame");
const sceneLink = document.getElementById("sceneLink");
let sceneKeyWindow = null;

function resolveAppUrl(path) {
  return new URL(path, appBaseUrl).toString();
}

function updatePorts(payload) {
  void payload;
}

function isEditableTarget(event) {
  const target = event.target;
  if (!target) return false;
  const tag = target.tagName ? target.tagName.toLowerCase() : "";
  return target.isContentEditable || ["input", "select", "textarea", "button"].includes(tag);
}

async function triggerTrack() {
  triggerButton.classList.add("active");
  setTimeout(() => triggerButton.classList.remove("active"), 140);
  try {
    const response = await fetch(resolveAppUrl("policy"), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({action: trackAction}),
    });
    const payload = await response.json();
    if (payload) updatePorts(payload);
    if (!response.ok) {
      policyStatus.textContent = `policy: ${trackKey.toUpperCase()} failed`;
      message.textContent = payload.error || "track request failed";
      return;
    }
    if (payload.sent) {
      policyStatus.textContent = `policy: ${trackKey.toUpperCase()} requested`;
      message.textContent = `Track requested via ${payload.action}.`;
    } else {
      policyStatus.textContent = `policy: ${trackKey.toUpperCase()} not delivered`;
      message.textContent = "Track request was not delivered.";
    }
  } catch (err) {
    policyStatus.textContent = `policy: ${trackKey.toUpperCase()} failed`;
    message.textContent = `track request failed: ${err}`;
  }
}

async function resetRollout(reason = "track_trigger_reset") {
  resetButton.classList.add("active");
  setTimeout(() => resetButton.classList.remove("active"), 140);
  try {
    const response = await fetch(resolveAppUrl("reset"), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({reason}),
    });
    const payload = await response.json();
    if (payload) updatePorts(payload);
    if (!response.ok) {
      policyStatus.textContent = `reset: ${resetKey.toUpperCase()} failed`;
      message.textContent = payload.error || "reset request failed";
      return;
    }
    policyStatus.textContent = `reset: ${resetKey.toUpperCase()} requested`;
    message.textContent = payload.policy_stopped
      ? "Rollout reset requested; policy stopped. Press S to start again."
      : "Rollout reset requested. Press S to start again.";
  } catch (err) {
    policyStatus.textContent = `reset: ${resetKey.toUpperCase()} failed`;
    message.textContent = `reset request failed: ${err}`;
  }
}

function handleKeyDown(event) {
  if (event.repeat || isEditableTarget(event)) return;
  const key = event.key.toLowerCase();
  if (key === trackKey) {
    event.preventDefault();
    triggerTrack();
    return;
  }
  if (key === resetKey || key === "backspace") {
    event.preventDefault();
    resetRollout(key === "backspace" ? "track_trigger_backspace_reset" : "track_trigger_reset");
  }
}

function attachSceneKeyHandlers() {
  if (!sceneFrame || !sceneFrame.contentWindow || sceneFrame.contentWindow === sceneKeyWindow) return;
  try {
    sceneKeyWindow = sceneFrame.contentWindow;
    sceneKeyWindow.addEventListener("keydown", handleKeyDown, true);
  } catch (err) {
    console.debug("scene key handlers unavailable", err);
  }
}

document.addEventListener("keydown", handleKeyDown, true);

triggerButton.addEventListener("click", () => triggerTrack());
resetButton.addEventListener("click", () => resetRollout("track_trigger_button_reset"));
sceneFrame.addEventListener("load", attachSceneKeyHandlers);

if (sceneUrl) {
  sceneFrame.src = sceneUrl;
  scenePanel.classList.add("active");
  sceneLink.innerHTML = `scene: <a href="${sceneUrl}" target="_blank" rel="noreferrer">${sceneUrl}</a>`;
}

fetch(resolveAppUrl("state"))
  .then((response) => response.json())
  .then((payload) => updatePorts(payload))
  .catch(() => {
    message.textContent = "state fetch failed";
  });
</script>
</body>
</html>
"""


def _filtered_proxy_headers(headers: Any) -> dict[str, str]:
    hop_by_hop = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-encoding",
        "content-length",
    }
    return {str(key): str(value) for key, value in headers.items() if str(key).lower() not in hop_by_hop}


def _upstream_url(base_url: str, tail: str, query: str, *, websocket: bool = False) -> str:
    split = urlsplit(base_url.rstrip("/") + "/")
    scheme = split.scheme
    if websocket:
        scheme = "wss" if scheme == "https" else "ws"
    path = "/" + tail.lstrip("/")
    return urlunsplit((scheme, split.netloc, path, query, ""))


async def _proxy_websocket(request: web.Request, scene_proxy_url: str, tail: str) -> web.WebSocketResponse:
    ws_client_response = web.WebSocketResponse()
    await ws_client_response.prepare(request)

    session: ClientSession = request.app["client_session"]
    upstream = _upstream_url(scene_proxy_url, tail, request.query_string, websocket=True)

    async with session.ws_connect(upstream, max_msg_size=0) as ws_upstream:
        async def browser_to_upstream() -> None:
            async for msg in ws_client_response:
                if msg.type == WSMsgType.TEXT:
                    await ws_upstream.send_str(msg.data)
                elif msg.type == WSMsgType.BINARY:
                    await ws_upstream.send_bytes(msg.data)
                elif msg.type == WSMsgType.ERROR:
                    break
            await ws_upstream.close()

        async def upstream_to_browser() -> None:
            async for msg in ws_upstream:
                if msg.type == WSMsgType.TEXT:
                    await ws_client_response.send_str(msg.data)
                elif msg.type == WSMsgType.BINARY:
                    await ws_client_response.send_bytes(msg.data)
                elif msg.type == WSMsgType.ERROR:
                    break
            await ws_client_response.close()

        await asyncio.gather(browser_to_upstream(), upstream_to_browser(), return_exceptions=True)
    return ws_client_response


async def _proxy_http(request: web.Request, scene_proxy_url: str, tail: str) -> web.Response:
    session: ClientSession = request.app["client_session"]
    upstream = _upstream_url(scene_proxy_url, tail, request.query_string)
    headers = _filtered_proxy_headers(request.headers)
    headers.pop("Host", None)
    data = await request.read() if request.can_read_body else None

    async with session.request(
        request.method,
        upstream,
        headers=headers,
        data=data,
        allow_redirects=False,
    ) as response:
        body = await response.read()
        response_headers = _filtered_proxy_headers(response.headers)
        return web.Response(status=response.status, headers=response_headers, body=body)


async def _scene_proxy_handler(request: web.Request) -> web.StreamResponse:
    scene_proxy_url = str(request.app["scene_proxy_url"])
    if not scene_proxy_url:
        return web.json_response({"error": "scene proxy disabled"}, status=404)

    tail = request.match_info.get("tail", "")
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return await _proxy_websocket(request, scene_proxy_url, tail)
    if not tail and not request.path.endswith("/"):
        raise web.HTTPFound("/scene/")
    return await _proxy_http(request, scene_proxy_url, tail)


def _resolve_action(action: str) -> tuple[str, list[str]]:
    action = str(action).strip().lower()
    if action in {"s", "track", "rollout_start", "start_rollout", "space_start", "start_with_motion"}:
        return "rollout_start", ["start", "space"]
    if action in {"]", "right_bracket", "start"}:
        return "start", ["start"]
    if action in {" ", "spacebar", "motion", "space", "start_motion", "start_motion_clip"}:
        return "space", ["space"]
    raise ValueError(f"unsupported policy action: {action}")


class TrackTriggerState:
    def __init__(self, sparse_root_command_port: int, control_port: int, policy_control_port: int) -> None:
        self.sparse_root_command_port = int(sparse_root_command_port)
        self.control_port = int(control_port)
        self.policy_control_port = int(policy_control_port)
        self.control_pub = SimControlPush(port=self.control_port)
        self.policy_pub = PolicyControlPush(port=self.policy_control_port) if self.policy_control_port > 0 else None

    def start(self) -> None:
        self.control_pub.start()
        if self.policy_pub is not None:
            self.policy_pub.start()

    def close(self) -> None:
        self.control_pub.close()
        if self.policy_pub is not None:
            self.policy_pub.close()

    def snapshot(self) -> dict[str, Any]:
        return {
            "sparse_root_command_port": self.sparse_root_command_port,
            "control_port": self.control_port,
            "policy_control_port": self.policy_control_port,
            "sim_control_enabled": bool(self.control_pub.enabled),
            "policy_control_enabled": bool(self.policy_pub and self.policy_pub.enabled),
        }

    def request_policy_sequence(self, actions: list[str], delay_s: float = 0.05) -> bool:
        if self.policy_pub is None:
            return False
        sent_all = True
        for idx, action in enumerate(actions):
            sent_all = self.policy_pub.publish(action, source="track_trigger_web") and sent_all
            if idx + 1 < len(actions):
                time.sleep(max(float(delay_s), 0.0))
        return sent_all

    def request_reset(self, reason: str, *, delay_s: float = 0.05) -> dict[str, Any]:
        policy_stopped = False
        if self.policy_pub is not None:
            policy_stopped = self.policy_pub.publish("stop", source="track_trigger_web_reset")
            time.sleep(max(float(delay_s), 0.0))
        self.control_pub.request_reset(str(reason))
        response = self.snapshot()
        response.update(
            {
                "ok": bool(self.control_pub.enabled),
                "sent": bool(self.control_pub.enabled),
                "reason": str(reason),
                "policy_stopped": bool(policy_stopped),
            }
        )
        return response


async def _create_app(args: argparse.Namespace, state: TrackTriggerState, index_html: str) -> web.Application:
    app = web.Application()
    app["state"] = state
    app["scene_proxy_url"] = str(args.scene_proxy_url or "")
    app["index_html"] = index_html
    app["default_track_action"] = str(args.track_action)

    async def on_startup(app_: web.Application) -> None:
        app_["client_session"] = ClientSession()

    async def on_cleanup(app_: web.Application) -> None:
        await app_["client_session"].close()

    async def index(_request: web.Request) -> web.Response:
        return web.Response(
            text=app["index_html"],
            content_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    async def state_route(_request: web.Request) -> web.Response:
        return web.json_response(state.snapshot())

    async def policy(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        raw_action = str(payload.get("action") or app["default_track_action"])
        try:
            canonical_action, sequence = _resolve_action(raw_action)
        except ValueError as exc:
            response = state.snapshot()
            response.update({"ok": False, "error": str(exc)})
            return web.json_response(response, status=400)
        sent = state.request_policy_sequence(sequence)
        response = state.snapshot()
        response.update({"ok": bool(sent), "sent": bool(sent), "action": canonical_action, "sequence": sequence})
        return web.json_response(response)

    async def reset(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        reason = str(payload.get("reason") or "track_trigger_reset")
        response = state.request_reset(reason)
        return web.json_response(response, status=200 if response.get("ok") else 503)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/", index)
    app.router.add_get("/index.html", index)
    app.router.add_get("/state", state_route)
    app.router.add_post("/policy", policy)
    app.router.add_post("/reset", reset)
    app.router.add_route("*", "/scene", _scene_proxy_handler)
    app.router.add_route("*", "/scene/{tail:.*}", _scene_proxy_handler)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Browser start/reset trigger for split MuJoCo tracking.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--sparse-root-command-port", type=int, default=5661)
    parser.add_argument("--control-port", type=int, default=5659)
    parser.add_argument("--policy-control-port", type=int, default=5662)
    # Accepted for compatibility with mj_env.sh's shared command-web args.
    parser.add_argument("--policy-overlay-port", type=int, default=5663)
    parser.add_argument("--scene-url", default="")
    parser.add_argument("--scene-proxy-url", default="")
    parser.add_argument("--track-key", default="s")
    parser.add_argument("--reset-key", default="r")
    parser.add_argument("--track-action", default="rollout_start")
    args = parser.parse_args()

    track_key = str(args.track_key or "s").strip().lower()
    if len(track_key) != 1:
        raise SystemExit(f"[ERROR] --track-key expects a single character, got: {args.track_key}")
    reset_key = str(args.reset_key or "r").strip().lower()
    if len(reset_key) != 1:
        raise SystemExit(f"[ERROR] --reset-key expects a single character, got: {args.reset_key}")
    if reset_key == track_key:
        raise SystemExit(f"[ERROR] --reset-key must differ from --track-key, got: {reset_key}")

    state = TrackTriggerState(
        sparse_root_command_port=args.sparse_root_command_port,
        control_port=args.control_port,
        policy_control_port=args.policy_control_port,
    )
    state.start()

    scene_url = str(args.scene_url or "")
    if args.scene_proxy_url and not scene_url:
        scene_url = "scene/"
    index_html = (
        INDEX_HTML.replace("__SCENE_URL_JSON__", json.dumps(scene_url))
        .replace("__TRACK_KEY_JSON__", json.dumps(track_key))
        .replace("__RESET_KEY_JSON__", json.dumps(reset_key))
        .replace("__TRACK_ACTION_JSON__", json.dumps(str(args.track_action)))
        .replace("__TRACK_KEY_LABEL__", track_key.upper())
        .replace("__RESET_KEY_LABEL__", reset_key.upper())
    )

    def _shutdown(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info("Open MuJoCo track trigger web at http://localhost:{}", args.port)
    if args.scene_proxy_url:
        logger.info("Proxying MuJoCo scene from {} at http://localhost:{}/scene/", args.scene_proxy_url, args.port)
    elif scene_url:
        logger.info("Showing MuJoCo scene iframe from {}", scene_url)
    logger.info(
        "Publishing track trigger on key '{}' with action '{}' via policy control port {}",
        track_key,
        args.track_action,
        args.policy_control_port,
    )
    logger.info("Publishing reset rollout on key '{}' via sim-control port {}", reset_key, args.control_port)
    try:
        app = asyncio.run(_create_app(args, state, index_html))
        web.run_app(app, host=args.host, port=int(args.port), print=None, handle_signals=False)
    except KeyboardInterrupt:
        logger.info("Stopping MuJoCo track trigger web")
    finally:
        state.close()


if __name__ == "__main__":
    main()
