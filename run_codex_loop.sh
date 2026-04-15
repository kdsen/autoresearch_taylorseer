#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/home/yjs/autoresearch}"
PROGRAM_MD="${PROGRAM_MD:-$WORKDIR/program.md}"
CONTINUE_PROMPT="${CONTINUE_PROMPT:-continue}"
SLEEP_SECONDS="${SLEEP_SECONDS:-3}"
CODEX_BIN="${CODEX_BIN:-codex}"
LOG_FILE="${LOG_FILE:-$WORKDIR/codex_loop.log}"
BLOCKER_STATUS=200

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  ./run_codex_loop.sh [extra codex exec args...]

Environment variables:
  WORKDIR=/home/yjs/autoresearch
  PROGRAM_MD=$WORKDIR/program.md
  CONTINUE_PROMPT=continue
  SLEEP_SECONDS=3
  CODEX_BIN=codex
  LOG_FILE=$WORKDIR/codex_loop.log

Examples:
  ./run_codex_loop.sh --full-auto
  LOG_FILE=/tmp/codex_loop.log ./run_codex_loop.sh --full-auto
USAGE
  exit 0
fi

EXTRA_ARGS=("$@")

if ! command -v "$CODEX_BIN" >/dev/null 2>&1; then
  echo "codex binary not found: $CODEX_BIN" >&2
  exit 1
fi

if [[ ! -d "$WORKDIR" ]]; then
  echo "workdir does not exist: $WORKDIR" >&2
  exit 1
fi

if [[ ! -f "$PROGRAM_MD" ]]; then
  echo "program.md not found: $PROGRAM_MD" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

output_has_blocker() {
  local output_file="$1"
  grep -Eiq \
    'bwrap: No permissions to create a new namespace|still cannot continue|no valid way to continue|blocked before any command starts|lacks write access|still cannot write|cannot write the Pad|Until shell execution works again|Until those environment constraints are fixed' \
    "$output_file"
}

run_codex_with_log() {
  local output_file="$1"
  shift
  "$@" 2>&1 | tee "$output_file"
}

stop_requested=0
on_signal() {
  stop_requested=1
  log "received stop signal; exiting after current step"
}
trap on_signal INT TERM

run_once() {
  cd "$WORKDIR"

  local resume_log
  local resume_status
  local bootstrap_log
  local bootstrap_status

  log "attempting resume from latest codex session"
  resume_log="$(mktemp)"
  if run_codex_with_log "$resume_log" "$CODEX_BIN" exec resume --last "${EXTRA_ARGS[@]}" "$CONTINUE_PROMPT"; then
    if output_has_blocker "$resume_log"; then
      resume_status=$BLOCKER_STATUS
      log "resume path reported a blocker despite zero exit; falling back to program.md bootstrap"
    else
      rm -f "$resume_log"
      log "resume path finished successfully"
      return 0
    fi
  else
    resume_status=$?
    log "resume path failed with status=$resume_status; falling back to program.md bootstrap"
  fi
  rm -f "$resume_log"

  bootstrap_log="$(mktemp)"
  if run_codex_with_log "$bootstrap_log" "$CODEX_BIN" exec "${EXTRA_ARGS[@]}" - < "$PROGRAM_MD"; then
    if output_has_blocker "$bootstrap_log"; then
      rm -f "$bootstrap_log"
      log "bootstrap path reported a blocker despite zero exit; stopping loop to avoid retrying a broken session"
      return "$BLOCKER_STATUS"
    fi
    rm -f "$bootstrap_log"
    log "bootstrap from program.md finished successfully"
    return 0
  fi

  bootstrap_status=$?
  if output_has_blocker "$bootstrap_log"; then
    rm -f "$bootstrap_log"
    log "bootstrap path failed due to an environment blocker; stopping loop to avoid retrying without a fix"
    return "$BLOCKER_STATUS"
  fi
  rm -f "$bootstrap_log"
  log "bootstrap from program.md failed with status=$bootstrap_status"
  return "$bootstrap_status"
}

attempt=1
log "starting codex loop in $WORKDIR"
log "using program file $PROGRAM_MD"
log "logging outer loop output to $LOG_FILE"

while [[ "$stop_requested" -eq 0 ]]; do
  log "loop attempt=$attempt"

  if run_once; then
    log "codex iteration completed"
  else
    status=$?
    log "codex iteration failed with status=$status"
    if [[ "$status" -eq "$BLOCKER_STATUS" ]]; then
      stop_requested=1
      log "environment blocker detected; stopping loop instead of retrying"
    fi
  fi

  if [[ "$stop_requested" -ne 0 ]]; then
    break
  fi

  log "sleeping for ${SLEEP_SECONDS}s before next attempt"
  sleep "$SLEEP_SECONDS"
  attempt=$((attempt + 1))
done

log "codex loop stopped"
