#!/usr/bin/env python3
"""Small browser control panel for split MuJoCo sparse-root commands."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

from aiohttp import ClientSession, WSMsgType, web
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[3]
INFER_SRC_ROOT = REPO_ROOT / "src" / "holosoma_inference"
for path in (REPO_ROOT / "src" / "holosoma", INFER_SRC_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from holosoma_inference.utils.sim_control import ManualRootCommandPub, PolicyControlPush, SimControlPush  # noqa: E402


CONTROL_KEYS = frozenset({"w", "s", "a", "d", "q", "e"})


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MuJoCo Command</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #15171b;
      color: #f4f4f5;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background:
        linear-gradient(120deg, rgba(31, 41, 55, 0.92), rgba(23, 26, 33, 0.98)),
        #15171b;
    }
    main {
      width: min(760px, calc(100vw - 32px));
      display: grid;
      gap: 16px;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .panel {
      border: 1px solid #2e3440;
      border-radius: 8px;
      background: #1f232b;
      padding: 18px;
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      max-width: 360px;
    }
    .key {
      height: 72px;
      border: 1px solid #3b4252;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: #252b35;
      color: #d8dee9;
      font-size: 24px;
      font-weight: 700;
      user-select: none;
    }
    .key.active {
      background: #2f7d5a;
      border-color: #63d297;
      color: white;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 12px;
      align-items: end;
    }
    label {
      display: grid;
      gap: 6px;
      color: #c7ccd6;
      font-size: 13px;
    }
    input, select, button {
      height: 40px;
      border-radius: 8px;
      border: 1px solid #3b4252;
      background: #171b22;
      color: #f4f4f5;
      font: inherit;
      padding: 0 12px;
    }
    input[type="checkbox"] {
      height: 18px;
      width: 18px;
      padding: 0;
      justify-self: start;
    }
    button {
      cursor: pointer;
      background: #26303d;
      font-weight: 600;
    }
    button:hover {
      border-color: #6b7280;
      background: #2e3948;
    }
    .status {
      display: grid;
      gap: 8px;
      color: #d8dee9;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 14px;
      overflow-wrap: anywhere;
    }
    .toolbar-inline {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
      gap: 8px;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 360px) minmax(260px, 1fr);
      gap: 16px;
      align-items: start;
    }
    .scene {
      display: none;
      padding: 0;
      overflow: hidden;
    }
    .scene.active {
      display: block;
    }
    iframe {
      width: 100%;
      height: min(62vh, 720px);
      min-height: 420px;
      border: 0;
      display: block;
      background: #101217;
    }
    .links {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      color: #c7ccd6;
      font-size: 14px;
    }
    a {
      color: #8bd5ff;
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
    @media (max-width: 720px) {
      .layout, .controls {
        grid-template-columns: 1fr;
      }
      .toolbar-inline {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
  </style>
</head>
<body>
<main>
  <h1>MuJoCo Command</h1>
  <section id="scenePanel" class="panel scene">
    <iframe id="sceneFrame" title="MuJoCo scene"></iframe>
  </section>
  <section class="panel layout">
    <div class="grid" aria-label="Keyboard command state">
      <div></div><div class="key" data-key="w">W</div><div></div>
      <div class="key" data-key="a">A</div><div class="key" data-key="s">S</div><div class="key" data-key="d">D</div>
      <div class="key" data-key="q">Q</div><div></div><div class="key" data-key="e">E</div>
    </div>
    <div class="status">
      <div id="state">command: [0.000, 0.000, 0.000]</div>
      <div id="policyStatus">policy: waiting for ]</div>
      <div id="ports"></div>
      <div class="toolbar-inline">
        <button id="policyRolloutStart" type="button">Space + ]</button>
        <button id="policyStart" type="button">Policy ]</button>
        <button id="policySpace" type="button">Policy Space</button>
        <button id="policyStop" type="button">Stop</button>
        <button id="policyInit" type="button">Init Pose</button>
      </div>
      <div id="sceneLink" class="links"></div>
      <div id="message">Using motion command.</div>
    </div>
  </section>
  <section class="panel controls">
    <label>Manual mode
      <input id="enabled" type="checkbox" />
    </label>
    <label>Default pose init
      <input id="resetToDefaultPose" type="checkbox" />
    </label>
    <label>XY value
      <input id="value" type="number" value="__VALUE_ATTR__" min="0" max="3" step="0.05" />
    </label>
    <label>Yaw deg
      <input id="yawDegrees" type="number" value="__YAW_DEGREES_ATTR__" min="0" max="180" step="1" />
    </label>
    <label>Mode
      <select id="mode">
        <option value="manual">manual</option>
        <option value="offset">offset</option>
      </select>
    </label>
    <button id="zero" type="button">Zero</button>
    <button id="reset" type="button">Reset Sim</button>
  </section>
</main>
<script>
const pressed = new Set();
const controlKeys = new Set(["w", "s", "a", "d", "q", "e"]);
const oppositeKey = {w: "s", s: "w", a: "d", d: "a", q: "e", e: "q"};
const enabled = document.getElementById("enabled");
const resetToDefaultPose = document.getElementById("resetToDefaultPose");
const value = document.getElementById("value");
const yawDegrees = document.getElementById("yawDegrees");
const mode = document.getElementById("mode");
const state = document.getElementById("state");
const policyStatus = document.getElementById("policyStatus");
const ports = document.getElementById("ports");
const message = document.getElementById("message");
const appBaseUrl = new URL("./", window.location.href);
function resolveAppUrl(path) {
  return new URL(path, appBaseUrl).toString();
}
const sceneUrlRaw = __SCENE_URL_JSON__;
const sceneUrl = sceneUrlRaw ? resolveAppUrl(sceneUrlRaw) : "";
const scenePanel = document.getElementById("scenePanel");
const sceneFrame = document.getElementById("sceneFrame");
const sceneLink = document.getElementById("sceneLink");
let sceneKeyWindow = null;
enabled.checked = __ENABLED_JSON__;
resetToDefaultPose.checked = __RESET_TO_DEFAULT_POSE_JSON__;
mode.value = __MODE_JSON__;
if (sceneUrl) {
  sceneFrame.src = sceneUrl;
  scenePanel.classList.add("active");
  sceneLink.innerHTML = `scene: <a href="${sceneUrl}" target="_blank" rel="noreferrer">${sceneUrl}</a>`;
}

function isEditableTarget(event) {
  const target = event.target;
  if (!target) return false;
  const tag = target.tagName ? target.tagName.toLowerCase() : "";
  return target.isContentEditable || ["input", "select", "textarea", "button"].includes(tag);
}

function updatePorts(payload) {
  const policyPort = payload.policy_control_enabled ? ` policy_control_port=${payload.policy_control_port}` : "";
  ports.textContent = `sparse_root_port=${payload.sparse_root_command_port} control_port=${payload.control_port}${policyPort}`;
}

function commandVector() {
  const scale = Math.abs(Number(value.value) || 0);
  const yawScale = Math.abs(Number(yawDegrees.value) || 0) * Math.PI / 180.0;
  const x = (pressed.has("w") ? scale : 0) + (pressed.has("s") ? -scale : 0);
  const y = (pressed.has("a") ? scale : 0) + (pressed.has("d") ? -scale : 0);
  const yaw = (pressed.has("q") ? yawScale : 0) + (pressed.has("e") ? -yawScale : 0);
  return [x, y, yaw];
}

function refreshKeys() {
  document.querySelectorAll(".key").forEach((el) => {
    el.classList.toggle("active", pressed.has(el.dataset.key));
  });
  const cmd = commandVector();
  state.textContent = `command: [${cmd.map((v) => v.toFixed(3)).join(", ")}]`;
}

async function sendCommand() {
  refreshKeys();
  const body = {
    keys: Array.from(pressed),
    enabled: enabled.checked,
    reset_to_default_pose: resetToDefaultPose.checked,
    value: Math.abs(Number(value.value) || 0),
    yaw_degrees: Math.abs(Number(yawDegrees.value) || 0),
    mode: mode.value,
  };
  try {
    const response = await fetch(resolveAppUrl("command"), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (payload && payload.command) {
      updatePorts(payload);
      if (!payload.publisher_enabled) {
        message.textContent = "Publisher is not bound.";
      } else if (payload.enabled) {
        message.textContent = "Publishing manual command.";
      } else {
        message.textContent = "Using motion command.";
      }
    }
  } catch (err) {
    message.textContent = `send failed: ${err}`;
  }
}

async function sendReset(reason = "web_command_reset") {
  pressed.clear();
  await sendCommand();
  try {
    const response = await fetch(resolveAppUrl("reset"), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        reason,
        reset_to_default_pose: resetToDefaultPose.checked,
      }),
    });
    const payload = await response.json();
    if (payload) updatePorts(payload);
    if (!response.ok) {
      message.textContent = payload.error || "reset failed";
      return;
    }
    message.textContent = `Reset requested (${payload.motion_init_mode || "unknown"}).`;
  } catch (err) {
    message.textContent = `reset failed: ${err}`;
  }
}

async function sendPolicy(action) {
  const label = policyActionLabel(action);
  try {
    const response = await fetch(resolveAppUrl("policy"), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({action}),
    });
    const payload = await response.json();
    if (payload) updatePorts(payload);
    if (!response.ok) {
      policyStatus.textContent = `policy: ${label} failed`;
      message.textContent = payload.error || `policy ${action} failed`;
    } else if (payload.sent) {
      policyStatus.textContent = `policy: ${label} requested`;
      message.textContent = `Policy ${label} requested.`;
    } else {
      policyStatus.textContent = `policy: ${label} not delivered`;
      message.textContent = `Policy ${label} was not delivered.`;
    }
  } catch (err) {
    policyStatus.textContent = `policy: ${label} failed`;
    message.textContent = `policy control failed: ${err}`;
  }
}

function toggleControlKey(key) {
  const opposite = oppositeKey[key];
  if (pressed.has(key)) {
    pressed.delete(key);
    return;
  }
  if (opposite) pressed.delete(opposite);
  pressed.add(key);
}

function policyActionLabel(action) {
  if (action === "rollout_start") return "Space + ]";
  if (action === "space") return "Space";
  if (action === "start") return "]";
  return action;
}

function handleKeyDown(event) {
  if (isEditableTarget(event)) return;
  const key = event.key.toLowerCase();
  if (!event.repeat && (key === "]" || event.code === "BracketRight")) {
    event.preventDefault();
    sendPolicy("start");
    return;
  }
  if (!event.repeat && (key === " " || key === "spacebar" || event.code === "Space")) {
    event.preventDefault();
    sendPolicy("space");
    return;
  }
  if (!event.repeat && key === "backspace") {
    event.preventDefault();
    sendReset("web_command_backspace_reset");
    return;
  }
  if (!controlKeys.has(key) || event.repeat) return;
  event.preventDefault();
  toggleControlKey(key);
  sendCommand();
}

function handleKeyUp(event) {
  const key = event.key.toLowerCase();
  if (!controlKeys.has(key)) return;
  event.preventDefault();
}

function attachSceneKeyHandlers() {
  if (!sceneFrame || !sceneFrame.contentWindow || sceneFrame.contentWindow === sceneKeyWindow) return;
  try {
    sceneKeyWindow = sceneFrame.contentWindow;
    sceneKeyWindow.addEventListener("keydown", handleKeyDown, true);
    sceneKeyWindow.addEventListener("keyup", handleKeyUp, true);
  } catch (err) {
    console.debug("scene key handlers unavailable", err);
  }
}

document.addEventListener("keydown", handleKeyDown, true);
document.addEventListener("keyup", handleKeyUp, true);
sceneFrame.addEventListener("load", attachSceneKeyHandlers);

enabled.addEventListener("change", sendCommand);
resetToDefaultPose.addEventListener("change", sendCommand);
value.addEventListener("change", sendCommand);
yawDegrees.addEventListener("change", sendCommand);
mode.addEventListener("change", sendCommand);
document.getElementById("zero").addEventListener("click", () => {
  pressed.clear();
  sendCommand();
});
document.getElementById("reset").addEventListener("click", async () => {
  await sendReset("web_command_reset");
});
document.getElementById("policyRolloutStart").addEventListener("click", () => sendPolicy("rollout_start"));
document.getElementById("policyStart").addEventListener("click", () => sendPolicy("start"));
document.getElementById("policySpace").addEventListener("click", () => sendPolicy("space"));
document.getElementById("policyStop").addEventListener("click", () => sendPolicy("stop"));
document.getElementById("policyInit").addEventListener("click", () => sendPolicy("init"));

setInterval(sendCommand, 100);
sendCommand();
</script>
</body>
</html>
"""


class CommandState:
    def __init__(
        self,
        sparse_root_command_port: int,
        control_port: int,
        policy_control_port: int,
        value: float,
        yaw_value: float,
        mode: str,
        enabled: bool,
        reset_to_default_pose: bool,
    ) -> None:
        self.sparse_root_command_port = int(sparse_root_command_port)
        self.control_port = int(control_port)
        self.policy_control_port = int(policy_control_port)
        self.value = abs(float(value))
        self.yaw_value = abs(float(yaw_value))
        self.mode = str(mode)
        self.enabled = bool(enabled)
        self.reset_to_default_pose = bool(reset_to_default_pose)
        self.keys: set[str] = set()
        self.lock = threading.Lock()
        self.pub = ManualRootCommandPub(port=self.sparse_root_command_port)
        self.control_pub = SimControlPush(port=self.control_port)
        self.policy_pub = PolicyControlPush(port=self.policy_control_port) if self.policy_control_port > 0 else None

    def start(self) -> None:
        self.pub.start()
        self.control_pub.start()
        if self.policy_pub is not None:
            self.policy_pub.start()
        self.publish_current()

    def close(self) -> None:
        self.pub.close()
        self.control_pub.close()
        if self.policy_pub is not None:
            self.policy_pub.close()

    def _command_locked(self) -> list[float]:
        value = abs(float(self.value))
        yaw_value = abs(float(self.yaw_value))
        x = (value if "w" in self.keys else 0.0) + (-value if "s" in self.keys else 0.0)
        y = (value if "a" in self.keys else 0.0) + (-value if "d" in self.keys else 0.0)
        yaw = (yaw_value if "q" in self.keys else 0.0) + (-yaw_value if "e" in self.keys else 0.0)
        return [float(x), float(y), float(yaw)]

    def update_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if "enabled" in payload:
                self.enabled = bool(payload["enabled"])
            if "reset_to_default_pose" in payload:
                self.reset_to_default_pose = bool(payload["reset_to_default_pose"])
            if "value" in payload:
                try:
                    self.value = abs(float(payload["value"]))
                except (TypeError, ValueError):
                    pass
            if "yaw_value" in payload:
                try:
                    self.yaw_value = abs(float(payload["yaw_value"]))
                except (TypeError, ValueError):
                    pass
            elif "yaw_degrees" in payload:
                try:
                    self.yaw_value = math.radians(abs(float(payload["yaw_degrees"])))
                except (TypeError, ValueError):
                    pass
            if "mode" in payload:
                mode = str(payload["mode"]).strip().lower()
                self.mode = mode if mode in {"manual", "offset"} else "manual"
            if "keys" in payload:
                raw_keys = payload.get("keys")
                if isinstance(raw_keys, list):
                    self.keys = {str(key).lower() for key in raw_keys if str(key).lower() in CONTROL_KEYS}
            elif "key" in payload:
                key = str(payload.get("key")).lower()
                down = bool(payload.get("down", False))
                if key in CONTROL_KEYS:
                    if down:
                        self.keys.add(key)
                    else:
                        self.keys.discard(key)
            command = self._command_locked()
            enabled = self.enabled
            mode = self.mode

        self.pub.publish(enabled=enabled, mode=mode, command=command)
        return self.snapshot(command=command)

    def publish_current(self) -> None:
        with self.lock:
            command = self._command_locked()
            enabled = self.enabled
            mode = self.mode
        self.pub.publish(enabled=enabled, mode=mode, command=command)

    def request_reset(self, reason: str, *, reset_to_default_pose: bool | None = None) -> dict[str, Any]:
        with self.lock:
            if reset_to_default_pose is not None:
                self.reset_to_default_pose = bool(reset_to_default_pose)
            motion_init_mode = "training_default_pose" if self.reset_to_default_pose else "raw_motion"
        self.control_pub.request_reset(reason, motion_init_mode=motion_init_mode)
        response = self.snapshot()
        response.update({"ok": True, "reason": str(reason), "motion_init_mode": motion_init_mode})
        return response

    def request_policy(self, action: str) -> bool:
        if self.policy_pub is None:
            return False
        return self.policy_pub.publish(action, source="web_command")

    def request_policy_sequence(self, actions: list[str], delay_s: float = 0.05) -> bool:
        sent_all = True
        for idx, action in enumerate(actions):
            sent_all = self.request_policy(action) and sent_all
            if idx + 1 < len(actions):
                time.sleep(max(float(delay_s), 0.0))
        return sent_all

    def snapshot(self, command: list[float] | None = None) -> dict[str, Any]:
        with self.lock:
            if command is None:
                command = self._command_locked()
            return {
                "enabled": self.enabled,
                "mode": self.mode,
                "value": self.value,
                "yaw_value": self.yaw_value,
                "yaw_degrees": math.degrees(self.yaw_value),
                "keys": sorted(self.keys),
                "command": command,
                "reset_to_default_pose": self.reset_to_default_pose,
                "publisher_enabled": bool(self.pub.enabled),
                "sparse_root_command_port": self.sparse_root_command_port,
                "control_port": self.control_port,
                "policy_control_port": self.policy_control_port,
                "policy_control_enabled": bool(self.policy_pub and self.policy_pub.enabled),
            }


def _heartbeat(state: CommandState, stop_event: threading.Event, rate_hz: float) -> None:
    period = 1.0 / max(float(rate_hz), 1.0)
    while not stop_event.wait(period):
        state.publish_current()


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


async def _create_app(args: argparse.Namespace, command_state: CommandState, index_html: str) -> web.Application:
    app = web.Application()
    app["command_state"] = command_state
    app["index_html"] = index_html
    app["scene_proxy_url"] = str(args.scene_proxy_url or "")

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

    async def state(_request: web.Request) -> web.Response:
        return web.json_response(command_state.snapshot())

    async def command(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        return web.json_response(command_state.update_from_payload(payload))

    async def reset(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        reason = str(payload.get("reason") or "web_command_reset")
        reset_to_default_pose = payload.get("reset_to_default_pose")
        return web.json_response(command_state.request_reset(reason, reset_to_default_pose=reset_to_default_pose))

    async def policy(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        action = str(payload.get("action") or "").strip().lower()
        if action in {"]", "right_bracket", "start"}:
            canonical_action = "start"
            sequence = ["start"]
        elif action in {" ", "spacebar", "motion", "start_motion", "start_motion_clip"}:
            canonical_action = "space"
            sequence = ["space"]
        elif action in {"rollout_start", "start_rollout", "space_start", "start_with_motion"}:
            canonical_action = "rollout_start"
            sequence = ["space", "start"]
        elif action in {"stop", "init", "space"}:
            canonical_action = action
            sequence = [action]
        else:
            response = command_state.snapshot()
            response.update({"ok": False, "error": f"unsupported policy action: {action}"})
            return web.json_response(response, status=400)
        try:
            sent = command_state.request_policy_sequence(sequence)
        except ValueError as exc:
            response = command_state.snapshot()
            response.update({"ok": False, "error": str(exc)})
            return web.json_response(response, status=400)
        response = command_state.snapshot()
        response.update({"ok": bool(sent), "sent": bool(sent), "action": canonical_action, "sequence": sequence})
        return web.json_response(response)

    async def zero(_request: web.Request) -> web.Response:
        return web.json_response(command_state.update_from_payload({"keys": []}))

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/", index)
    app.router.add_get("/index.html", index)
    app.router.add_get("/state", state)
    app.router.add_post("/command", command)
    app.router.add_post("/reset", reset)
    app.router.add_post("/policy", policy)
    app.router.add_post("/zero", zero)
    app.router.add_route("*", "/scene", _scene_proxy_handler)
    app.router.add_route("*", "/scene/{tail:.*}", _scene_proxy_handler)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Browser sparse-root command panel for split MuJoCo.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--sparse-root-command-port", type=int, default=5661)
    parser.add_argument("--control-port", type=int, default=5659)
    parser.add_argument("--policy-control-port", type=int, default=5662)
    parser.add_argument("--value", type=float, default=0.5)
    parser.add_argument("--yaw-degrees", type=float, default=17.0)
    parser.add_argument("--yaw-value", type=float, default=None, help="Yaw command step in radians; overrides --yaw-degrees.")
    parser.add_argument("--mode", choices=("manual", "offset"), default="manual")
    parser.add_argument("--scene-url", default="")
    parser.add_argument("--scene-proxy-url", default="")
    default_reset_to_default_pose = (
        os.environ.get("SIM_MOTION_INIT_MODE", "").strip().lower().replace("-", "_") == "training_default_pose"
    )
    reset_group = parser.add_mutually_exclusive_group()
    reset_group.add_argument("--reset-to-default-pose", dest="reset_to_default_pose", action="store_true")
    reset_group.add_argument("--no-reset-to-default-pose", dest="reset_to_default_pose", action="store_false")
    enabled_group = parser.add_mutually_exclusive_group()
    enabled_group.add_argument("--enabled", dest="enabled", action="store_true", help="Start with manual command enabled.")
    enabled_group.add_argument("--no-enabled", dest="enabled", action="store_false", help="Start with manual command disabled.")
    parser.set_defaults(enabled=False, reset_to_default_pose=default_reset_to_default_pose)
    parser.add_argument("--publish-rate-hz", type=float, default=20.0)
    args = parser.parse_args()
    yaw_value = (
        abs(float(args.yaw_value))
        if args.yaw_value is not None
        else math.radians(abs(float(args.yaw_degrees)))
    )

    command_state = CommandState(
        sparse_root_command_port=args.sparse_root_command_port,
        control_port=args.control_port,
        policy_control_port=args.policy_control_port,
        value=args.value,
        yaw_value=yaw_value,
        mode=args.mode,
        enabled=bool(args.enabled),
        reset_to_default_pose=bool(args.reset_to_default_pose),
    )
    command_state.start()

    stop_event = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat,
        args=(command_state, stop_event, float(args.publish_rate_hz)),
        daemon=True,
    )
    heartbeat.start()

    scene_url = str(args.scene_url or "")
    if args.scene_proxy_url and not scene_url:
        scene_url = "scene/"
    index_html = (
        INDEX_HTML.replace("__SCENE_URL_JSON__", json.dumps(scene_url))
        .replace("__VALUE_ATTR__", f"{abs(float(args.value)):.6g}")
        .replace("__YAW_DEGREES_ATTR__", f"{math.degrees(yaw_value):.6g}")
        .replace("__ENABLED_JSON__", json.dumps(bool(args.enabled)))
        .replace("__RESET_TO_DEFAULT_POSE_JSON__", json.dumps(bool(args.reset_to_default_pose)))
        .replace("__MODE_JSON__", json.dumps(str(args.mode)))
    )

    def _shutdown(_signum: int, _frame: object) -> None:
        stop_event.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info("Open MuJoCo command web at http://localhost:{}", args.port)
    if args.scene_proxy_url:
        logger.info("Proxying MuJoCo scene from {} at http://localhost:{}/scene/", args.scene_proxy_url, args.port)
    elif scene_url:
        logger.info("Showing MuJoCo scene iframe from {}", scene_url)
    logger.info(
        "Publishing sparse-root command on port {}, reset control on port {}, policy control on port {}, manual enabled={}, reset_to_default_pose={}, xy value {:.3f}, yaw {:.3f} rad ({:.1f} deg)",
        args.sparse_root_command_port,
        args.control_port,
        args.policy_control_port,
        bool(args.enabled),
        bool(args.reset_to_default_pose),
        abs(float(args.value)),
        yaw_value,
        math.degrees(yaw_value),
    )
    try:
        app = asyncio.run(_create_app(args, command_state, index_html))
        web.run_app(app, host=args.host, port=int(args.port), print=None, handle_signals=False)
    except KeyboardInterrupt:
        logger.info("Stopping MuJoCo command web")
    finally:
        stop_event.set()
        command_state.close()


if __name__ == "__main__":
    main()
