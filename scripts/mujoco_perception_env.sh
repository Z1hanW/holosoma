#!/usr/bin/env bash

# Preserve whether a MuJoCo perception value came from the user or from a
# launcher fallback.  Checkpoint metadata may replace launcher fallbacks, but
# must never replace an explicit user value.

holosoma_capture_explicit_env() {
  local name marker raw normalized
  for name in "$@"; do
    marker="${name}_EXPLICIT"
    raw=""
    if [[ -v "$marker" ]]; then
      raw="${!marker}"
    fi
    case "$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')" in
      1|true|yes|on)
        normalized=1
        ;;
      0|false|no|off)
        normalized=0
        ;;
      "")
        normalized=0
        if [[ -v "$name" && -n "${!name}" ]]; then
          normalized=1
        fi
        ;;
      *)
        echo "[ERROR] ${marker} must be a boolean marker (0/1/False/True), got: ${raw}" >&2
        return 2
        ;;
    esac
    printf -v "$marker" '%s' "$normalized"
    export "$marker"
  done
}

holosoma_set_launcher_default() {
  local name="$1"
  local default_value="$2"
  holosoma_capture_explicit_env "$name" || return
  if [[ ! -v "$name" || -z "${!name}" ]]; then
    printf -v "$name" '%s' "$default_value"
    export "$name"
  fi
}

holosoma_apply_checkpoint_value() {
  local name="$1"
  local checkpoint_value="$2"
  local marker="${name}_EXPLICIT"
  holosoma_capture_explicit_env "$name" || return
  if [[ "${!marker}" != "1" ]]; then
    printf -v "$name" '%s' "$checkpoint_value"
    export "$name"
  fi
}
