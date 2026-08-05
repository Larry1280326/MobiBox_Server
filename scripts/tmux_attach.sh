#!/bin/bash
# MobiBox Backend — Attach to tmux Session
#
# Usage:
#   ./scripts/tmux_attach.sh            # Attach to session
#   ./scripts/tmux_attach.sh --readonly # Attach in read-only mode

SESSION="mobibox"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if ! command -v tmux &> /dev/null; then
    echo -e "${RED}Error: tmux is not installed.${NC}"
    exit 1
fi

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo -e "${RED}No tmux session '$SESSION' found.${NC}"
    echo -e "${YELLOW}Start services first:${NC}  ./scripts/tmux_start.sh"
    exit 1
fi

if [[ "$1" == "--readonly" ]]; then
    tmux attach -t "$SESSION" -r
else
    tmux attach -t "$SESSION"
fi
