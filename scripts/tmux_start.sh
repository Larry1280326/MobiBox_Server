#!/bin/bash
# MobiBox Backend — Start All Services in tmux Session
#
# Creates a tmux session "mobibox" with separate windows for each service.
# Python's RotatingFileHandler handles log rotation (10 MB / 5 backups).
#
# Usage:
#   ./scripts/tmux_start.sh              # Start fresh
#   ./scripts/tmux_start.sh --attach     # Start and immediately attach
#   ./scripts/tmux_start.sh --no-docker  # Skip Docker container checks
#
# After starting, attach with:  tmux attach -t mobibox
# Or use:                        ./scripts/tmux_attach.sh

set -e

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── Config ──────────────────────────────────────────────────────────────────
SESSION="mobibox"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/logs"

mkdir -p "$LOGS_DIR"

ATTACH=false
SKIP_DOCKER=false
for arg in "$@"; do
    case $arg in
        --attach) ATTACH=true ;;
        --no-docker) SKIP_DOCKER=true ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  MobiBox Backend — tmux Startup${NC}"
echo -e "${BLUE}========================================${NC}"

# ── Pre-flight checks ──────────────────────────────────────────────────────

if ! command -v tmux &> /dev/null; then
    echo -e "${RED}Error: tmux is not installed.${NC}"
    echo -e "${YELLOW}Install with: sudo apt install tmux   (Debian/Ubuntu)${NC}"
    echo -e "${YELLOW}            sudo yum install tmux   (RHEL/CentOS)${NC}"
    exit 1
fi

if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    exit 1
fi

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate Mobibox_backend 2>/dev/null || {
    echo -e "${RED}Error: Failed to activate conda environment 'Mobibox_backend'${NC}"
    exit 1
}

# Check .env file
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${RED}Error: .env file not found at $PROJECT_ROOT/.env${NC}"
    echo -e "${YELLOW}Copy .env.example to .env and configure your credentials.${NC}"
    exit 1
fi

# ── Kill existing session if present ───────────────────────────────────────

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo -e "${YELLOW}Existing tmux session '$SESSION' found. Killing it...${NC}"
    tmux kill-session -t "$SESSION"
    sleep 1
fi

# ── Docker containers (optional) ───────────────────────────────────────────

if [ "$SKIP_DOCKER" = false ]; then
    echo -e "${BLUE}Checking Docker containers...${NC}"

    # MongoDB
    if docker ps --format '{{.Names}}' | grep -q '^mobibox-mongo$'; then
        echo -e "  ${GREEN}✓${NC} MongoDB container running"
    elif docker ps -a --format '{{.Names}}' | grep -q '^mobibox-mongo$'; then
        echo -e "  ${YELLOW}Starting existing MongoDB container...${NC}"
        docker start mobibox-mongo > /dev/null 2>&1
    else
        echo -e "  ${YELLOW}Creating MongoDB container...${NC}"
        docker run -d --name mobibox-mongo \
            -p 27017:27017 \
            -v mobibox_mongo_data:/data/db \
            mongo:7 > /dev/null 2>&1 || echo -e "  ${YELLOW}⚠ MongoDB may already be running externally${NC}"
    fi

    # RabbitMQ
    if docker ps --format '{{.Names}}' | grep -q '^rabbitmq$'; then
        echo -e "  ${GREEN}✓${NC} RabbitMQ container running"
    elif docker ps -a --format '{{.Names}}' | grep -q '^rabbitmq$'; then
        echo -e "  ${YELLOW}Starting existing RabbitMQ container...${NC}"
        docker start rabbitmq > /dev/null 2>&1
    else
        echo -e "  ${YELLOW}Creating RabbitMQ container...${NC}"
        docker run -d --name rabbitmq \
            -p 5672:5672 -p 15672:15672 \
            rabbitmq:3-management > /dev/null 2>&1 || echo -e "  ${YELLOW}⚠ RabbitMQ may already be running externally${NC}"
    fi
    echo ""
fi

# ── Create tmux session ────────────────────────────────────────────────────

echo -e "${BLUE}Creating tmux session '$SESSION'...${NC}"
cd "$PROJECT_ROOT"

# Create session with first window (detached)
tmux new-session -d -s "$SESSION" -n "api" -c "$PROJECT_ROOT"

# Configure tmux options for this session
tmux set-option -t "$SESSION" -g mouse on
tmux set-option -t "$SESSION" -g history-limit 50000
tmux set-option -t "$SESSION" -g remain-on-exit on
tmux set-option -t "$SESSION" -g set-titles on
tmux set-option -t "$SESSION" -g pane-border-status top

# ── Window 0: FastAPI ──────────────────────────────────────────────────────

tmux send-keys -t "$SESSION:api" \
    "echo -e '${BLUE}=== FastAPI Server (uvicorn) ===${NC}'" Enter
tmux send-keys -t "$SESSION:api" \
    "echo -e '${GREEN}Logs → $LOGS_DIR/api.log${NC}'" Enter
tmux send-keys -t "$SESSION:api" \
    "echo -e '${GREEN}Health → curl http://localhost:8001/health${NC}'" Enter
tmux send-keys -t "$SESSION:api" \
    "echo ''" Enter
tmux send-keys -t "$SESSION:api" \
    "eval \"\$(conda shell.bash hook)\" && conda activate Mobibox_backend && echo '[conda] Mobibox_backend activated'" Enter
tmux send-keys -t "$SESSION:api" \
    "uvicorn src.main:app --host 0.0.0.0 --port 8001 --log-level info 2>&1" Enter

# ── Window 1: Celery Worker ────────────────────────────────────────────────

tmux new-window -t "$SESSION" -n "worker" -c "$PROJECT_ROOT"
tmux send-keys -t "$SESSION:worker" \
    "echo -e '${BLUE}=== Celery Worker ===${NC}'" Enter
tmux send-keys -t "$SESSION:worker" \
    "echo -e '${GREEN}Logs → $LOGS_DIR/celery_worker.log${NC}'" Enter
tmux send-keys -t "$SESSION:worker" \
    "echo ''" Enter
tmux send-keys -t "$SESSION:worker" \
    "eval \"\$(conda shell.bash hook)\" && conda activate Mobibox_backend && echo '[conda] Mobibox_backend activated'" Enter
tmux send-keys -t "$SESSION:worker" \
    "celery -A src.celery_app.celery_app worker --loglevel=info -Q default,har,atomic,summary 2>&1" Enter

# ── Window 2: Celery Beat ──────────────────────────────────────────────────

tmux new-window -t "$SESSION" -n "beat" -c "$PROJECT_ROOT"
tmux send-keys -t "$SESSION:beat" \
    "echo -e '${BLUE}=== Celery Beat (Scheduler) ===${NC}'" Enter
tmux send-keys -t "$SESSION:beat" \
    "echo -e '${GREEN}Logs → $LOGS_DIR/celery_beat.log${NC}'" Enter
tmux send-keys -t "$SESSION:beat" \
    "echo ''" Enter
tmux send-keys -t "$SESSION:beat" \
    "eval \"\$(conda shell.bash hook)\" && conda activate Mobibox_backend && echo '[conda] Mobibox_backend activated'" Enter
tmux send-keys -t "$SESSION:beat" \
    "celery -A src.celery_app.celery_app beat --loglevel=info 2>&1" Enter

# ── Window 3: Live Log Monitor ─────────────────────────────────────────────

tmux new-window -t "$SESSION" -n "logs" -c "$PROJECT_ROOT"
tmux send-keys -t "$SESSION:logs" \
    "eval \"\$(conda shell.bash hook)\" && conda activate Mobibox_backend" Enter
tmux send-keys -t "$SESSION:logs" \
    "echo -e '${BLUE}=== Live Log Monitor ===${NC}'" Enter
tmux send-keys -t "$SESSION:logs" \
    "echo -e '${YELLOW}Ctrl-C to stop watching; services keep running in other windows${NC}'" Enter
tmux send-keys -t "$SESSION:logs" \
    "echo ''" Enter

# Split the logs window: top = api, bottom-left = worker, bottom-right = beat
tmux split-window -t "$SESSION:logs" -v -p 60 -c "$PROJECT_ROOT"
tmux send-keys -t "$SESSION:logs.1" \
    "tail -f $LOGS_DIR/celery_worker.log 2>/dev/null || echo 'Waiting for worker log...'" Enter

tmux split-window -t "$SESSION:logs.1" -h -c "$PROJECT_ROOT"
tmux send-keys -t "$SESSION:logs.2" \
    "tail -f $LOGS_DIR/celery_beat.log 2>/dev/null || echo 'Waiting for beat log...'" Enter

# Top pane: API logs
tmux select-pane -t "$SESSION:logs.0"
tmux send-keys -t "$SESSION:logs.0" \
    "tail -f $LOGS_DIR/api.log 2>/dev/null || echo 'Waiting for API log...'" Enter

# Return to the API window as default
tmux select-window -t "$SESSION:api"

# ── Done ────────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  All services started in tmux!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Session:${NC}  tmux attach -t $SESSION"
echo -e "          ./scripts/tmux_attach.sh"
echo ""
echo -e "${BLUE}Windows:${NC}"
echo -e "  ${GREEN}api${NC}     — FastAPI server (port 8001)"
echo -e "  ${GREEN}worker${NC}  — Celery worker"
echo -e "  ${GREEN}beat${NC}    — Celery beat scheduler"
echo -e "  ${GREEN}logs${NC}    — Live log monitor"
echo ""
echo -e "${BLUE}Navigation:${NC}"
echo -e "  Ctrl-b 0..3    — Switch windows"
echo -e "  Ctrl-b n / p   — Next / previous window"
echo -e "  Ctrl-b d       — Detach (services keep running)"
echo -e "  Ctrl-b [       — Scroll mode (q to quit)"
echo ""
echo -e "${BLUE}Log Rotation:${NC}"
echo -e "  Python RotatingFileHandler: 10 MB per file, 5 backups"
echo -e "  Logs: $LOGS_DIR/"
echo ""
echo -e "${BLUE}Stop:${NC}  ./scripts/tmux_stop.sh"
echo -e "${BLUE}Status:${NC} ./scripts/tmux_status.sh"
echo ""

if [ "$ATTACH" = true ]; then
    tmux attach -t "$SESSION"
fi
