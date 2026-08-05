#!/bin/bash
# MobiBox Backend — tmux Service Status
#
# Usage:
#   ./scripts/tmux_status.sh

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SESSION="mobibox"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/logs"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  MobiBox Backend — tmux Status${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ── tmux Session ────────────────────────────────────────────────────────────

echo -e "${BLUE}tmux Session:${NC}"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Session '$SESSION' is active"

    # List windows
    echo ""
    echo -e "  ${BLUE}Windows:${NC}"
    tmux list-windows -t "$SESSION" -F "    #{window_index}: #{window_name}#{?window_active, (active),}" 2>/dev/null | while read line; do
        echo -e "    ${GREEN}$line${NC}"
    done
else
    echo -e "  ${RED}✗${NC} Session '$SESSION' is NOT running"
fi
echo ""

# ── FastAPI ─────────────────────────────────────────────────────────────────

echo -e "${BLUE}FastAPI Server (port 8001):${NC}"
if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    HEALTH=$(curl -s http://localhost:8001/health 2>/dev/null)
    echo -e "  ${GREEN}✓${NC} Running — $HEALTH"
else
    if pgrep -f "uvicorn src.main:app" > /dev/null 2>&1; then
        echo -e "  ${YELLOW}⚠${NC} Process running but not responding"
    else
        echo -e "  ${RED}✗${NC} Not running"
    fi
fi
echo ""

# ── Celery Worker ──────────────────────────────────────────────────────────

echo -e "${BLUE}Celery Worker:${NC}"
if pgrep -f "celery.*worker" > /dev/null 2>&1; then
    WORKER_COUNT=$(pgrep -f "celery.*worker" | wc -l)
    echo -e "  ${GREEN}✓${NC} Running ($WORKER_COUNT process(es))"
    # Try to get registered tasks
    TASKS=$(celery -A src.celery_app.celery_app inspect registered 2>/dev/null | head -5 || true)
    if [ -n "$TASKS" ]; then
        echo -e "    Tasks registered: $(echo "$TASKS" | grep -c '\.' || true)"
    fi
else
    echo -e "  ${RED}✗${NC} Not running"
fi
echo ""

# ── Celery Beat ────────────────────────────────────────────────────────────

echo -e "${BLUE}Celery Beat:${NC}"
if pgrep -f "celery.*beat" > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Running"
else
    echo -e "  ${RED}✗${NC} Not running"
fi
echo ""

# ── Docker Containers ──────────────────────────────────────────────────────

echo -e "${BLUE}Docker Containers:${NC}"

# MongoDB
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^mobibox-mongo$'; then
    echo -e "  ${GREEN}✓${NC} MongoDB (mobibox-mongo)"
elif docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^mobibox-mongo$'; then
    echo -e "  ${YELLOW}⚠${NC} MongoDB stopped (container exists)"
else
    echo -e "  ${YELLOW}—${NC} MongoDB not managed by Docker"
fi

# RabbitMQ
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^rabbitmq$'; then
    echo -e "  ${GREEN}✓${NC} RabbitMQ (rabbitmq)"
elif docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^rabbitmq$'; then
    echo -e "  ${YELLOW}⚠${NC} RabbitMQ stopped (container exists)"
else
    echo -e "  ${YELLOW}—${NC} RabbitMQ not managed by Docker"
fi
echo ""

# ── Log Files ──────────────────────────────────────────────────────────────

echo -e "${BLUE}Log Files (with rotation):${NC}"
for logfile in api.log celery_worker.log celery_beat.log; do
    LOGPATH="$LOGS_DIR/$logfile"
    if [ -f "$LOGPATH" ]; then
        SIZE=$(du -h "$LOGPATH" | cut -f1)
        LINES=$(wc -l < "$LOGPATH" 2>/dev/null || echo "?")
        echo -e "  ${GREEN}✓${NC} $logfile — $SIZE, $LINES lines"
    else
        echo -e "  ${YELLOW}—${NC} $logfile (not created yet)"
    fi
done

# Show backup files if any
BACKUPS=$(ls "$LOGS_DIR"/*.log.* 2>/dev/null | wc -l || true)
if [ "$BACKUPS" -gt 0 ]; then
    echo -e "  ${BLUE}  + $BACKUPS rotated backup file(s)${NC}"
fi
echo ""

# ── Quick Commands ─────────────────────────────────────────────────────────

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Quick Commands${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "  Attach:        ./scripts/tmux_attach.sh"
echo -e "  Stop:          ./scripts/tmux_stop.sh"
echo -e "  Restart:       ./scripts/tmux_stop.sh && ./scripts/tmux_start.sh"
echo -e "  View API logs: tail -f $LOGS_DIR/api.log"
echo -e "  Test API:      curl http://localhost:8001/health"
echo ""
