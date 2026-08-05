#!/bin/bash
# MobiBox Backend — Stop All tmux-Managed Services
#
# Kills the tmux session, which terminates all services running inside it.
#
# Usage:
#   ./scripts/tmux_stop.sh             # Stop services, remove session
#   ./scripts/tmux_stop.sh --force     # Skip confirmation prompt

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SESSION="mobibox"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/logs"

FORCE=false
[[ "$1" == "--force" ]] && FORCE=true

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  MobiBox Backend — tmux Shutdown${NC}"
echo -e "${BLUE}========================================${NC}"

# ── Check if session exists ─────────────────────────────────────────────────

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo -e "${YELLOW}No tmux session '$SESSION' found.${NC}"
    echo ""

    # Still check for orphaned processes
    echo -e "${BLUE}Checking for orphaned processes...${NC}"
    FOUND=false

    if pgrep -f "uvicorn src.main:app" > /dev/null 2>&1; then
        echo -e "  ${YELLOW}Found orphaned FastAPI process${NC}"
        FOUND=true
    fi
    if pgrep -f "celery.*worker" > /dev/null 2>&1; then
        echo -e "  ${YELLOW}Found orphaned Celery worker${NC}"
        FOUND=true
    fi
    if pgrep -f "celery.*beat" > /dev/null 2>&1; then
        echo -e "  ${YELLOW}Found orphaned Celery beat${NC}"
        FOUND=true
    fi

    if [ "$FOUND" = true ]; then
        echo ""
        echo -e "${YELLOW}Kill orphaned processes? (y/N)${NC}"
        if [ "$FORCE" = true ]; then
            REPLY="y"
            echo "y (--force)"
        else
            read -r REPLY
        fi
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pkill -f "uvicorn src.main:app" 2>/dev/null || true
            pkill -f "celery.*worker" 2>/dev/null || true
            pkill -f "celery.*beat" 2>/dev/null || true
            sleep 1
            echo -e "${GREEN}✓ Orphaned processes killed${NC}"
        fi
    else
        echo -e "  ${GREEN}✓ No orphaned processes${NC}"
    fi

    rm -f "$LOGS_DIR"/*.pid 2>/dev/null || true
    echo ""
    echo -e "${BLUE}To start services:${NC}  ./scripts/tmux_start.sh"
    echo ""
    exit 0
fi

# ── Confirmation ────────────────────────────────────────────────────────────

if [ "$FORCE" != true ]; then
    echo ""
    echo -e "${YELLOW}This will kill the tmux session '$SESSION' and all services inside it.${NC}"
    echo -e "${YELLOW}Continue? (y/N)${NC}"
    read -r REPLY
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Aborted. Services still running.${NC}"
        exit 0
    fi
    echo ""
fi

# ── Kill tmux session ──────────────────────────────────────────────────────

echo -e "${YELLOW}Killing tmux session '$SESSION'...${NC}"
tmux kill-session -t "$SESSION"
sleep 1

# ── Verify cleanup ─────────────────────────────────────────────────────────

echo ""
echo -e "${BLUE}Verifying cleanup...${NC}"

REMAINING=false
if pgrep -f "uvicorn src.main:app" > /dev/null 2>&1; then
    echo -e "  ${RED}✗${NC} FastAPI process still running"
    REMAINING=true
else
    echo -e "  ${GREEN}✓${NC} FastAPI stopped"
fi
if pgrep -f "celery.*worker" > /dev/null 2>&1; then
    echo -e "  ${RED}✗${NC} Celery worker still running"
    REMAINING=true
else
    echo -e "  ${GREEN}✓${NC} Celery worker stopped"
fi
if pgrep -f "celery.*beat" > /dev/null 2>&1; then
    echo -e "  ${RED}✗${NC} Celery beat still running"
    REMAINING=true
else
    echo -e "  ${GREEN}✓${NC} Celery beat stopped"
fi

# Force kill if any remain
if [ "$REMAINING" = true ]; then
    echo ""
    echo -e "${YELLOW}Force-killing remaining processes...${NC}"
    pkill -9 -f "uvicorn src.main:app" 2>/dev/null || true
    pkill -9 -f "celery.*worker" 2>/dev/null || true
    pkill -9 -f "celery.*beat" 2>/dev/null || true
    sleep 1
    echo -e "${GREEN}✓ Done${NC}"
fi

# ── Clean up PID files ─────────────────────────────────────────────────────

rm -f "$LOGS_DIR"/*.pid 2>/dev/null || true

# ── Docker (optional) ──────────────────────────────────────────────────────

echo ""
echo -e "${YELLOW}Stop Docker containers? (y/N)${NC}"
echo -e "  (MongoDB: mobibox-mongo, RabbitMQ: rabbitmq)"
if [ "$FORCE" = true ]; then
    echo "n (--force skips this)"
else
    read -r REPLY
fi
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker stop rabbitmq 2>/dev/null && echo -e "  ${GREEN}✓${NC} RabbitMQ stopped" || true
    docker stop mobibox-mongo 2>/dev/null && echo -e "  ${GREEN}✓${NC} MongoDB stopped" || true
else
    echo -e "  ${YELLOW}Docker containers left running${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Services stopped${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}To restart:${NC}  ./scripts/tmux_start.sh"
echo ""
