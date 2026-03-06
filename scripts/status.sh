#!/bin/bash
# MobiBox Backend - Check Service Status
# This script checks the status of all MobiBox backend services

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  MobiBox Backend - Service Status${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# =========================================
# Check RabbitMQ
# =========================================
echo -e "${BLUE}RabbitMQ:${NC}"
if docker ps | grep -q rabbitmq; then
    echo -e "  ${GREEN}✓${NC} Running (Docker container)"
    echo -e "    Web UI: http://localhost:15672 (guest/guest)"
else
    if docker ps -a | grep -q rabbitmq; then
        echo -e "  ${YELLOW}⚠${NC} Stopped (container exists)"
    else
        echo -e "  ${RED}✗${NC} Not found"
    fi
fi
echo ""

# =========================================
# Check FastAPI
# =========================================
echo -e "${BLUE}FastAPI Server:${NC}"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Running"
    echo -e "    URL: http://localhost:8000"
    echo -e "    Health: $(curl -s http://localhost:8000/health 2>/dev/null || echo 'Unknown')"
else
    if pgrep -f "uvicorn src.main:app" > /dev/null; then
        echo -e "  ${YELLOW}⚠${NC} Process running but not responding"
    else
        echo -e "  ${RED}✗${NC} Not running"
    fi
fi
echo ""

# =========================================
# Check Celery Worker
# =========================================
echo -e "${BLUE}Celery Worker:${NC}"
if pgrep -f "celery.*worker" > /dev/null; then
    echo -e "  ${GREEN}✓${NC} Running"
    # Try to get worker status
    if command -v celery &> /dev/null; then
        echo -e "    Inspect registered tasks:"
        echo -e "    celery -A src.celery_app.celery_app inspect registered"
    fi
else
    echo -e "  ${RED}✗${NC} Not running"
fi
echo ""

# =========================================
# Check Celery Beat
# =========================================
echo -e "${BLUE}Celery Beat:${NC}"
if pgrep -f "celery.*beat" > /dev/null; then
    echo -e "  ${GREEN}✓${NC} Running"
else
    echo -e "  ${RED}✗${NC} Not running"
fi
echo ""

# =========================================
# Helper Function (must be defined before use)
# =========================================
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# =========================================
# Check TSFM Model
# =========================================
echo -e "${BLUE}TSFM Model:${NC}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TSFM_CHECKPOINT="$PROJECT_ROOT/src/celery_app/services/tsfm_model/ckpts/best.pt"
if [ -f "$TSFM_CHECKPOINT" ]; then
    SIZE=$(du -h "$TSFM_CHECKPOINT" | cut -f1)
    echo -e "  ${GREEN}✓${NC} Checkpoint found ($SIZE)"
else
    echo -e "  ${YELLOW}⚠${NC} Checkpoint not found"
    echo -e "    Download from remote server to: $TSFM_CHECKPOINT"
fi
echo ""

# =========================================
# Check Ports
# =========================================
echo -e "${BLUE}Port Usage:${NC}"
echo -e "  5672  (RabbitMQ):    $(check_port 5672 && echo -e "${GREEN}In Use${NC}" || echo -e "${RED}Free${NC}")"
echo -e "  8000  (FastAPI):     $(check_port 8000 && echo -e "${GREEN}In Use${NC}" || echo -e "${RED}Free${NC}")"
echo ""

# =========================================
# Summary
# =========================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Quick Commands${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "  Start all:    ./scripts/start_services.sh"
echo -e "  Stop all:     ./scripts/stop_services.sh"
echo -e "  Restart all:   ./scripts/restart_services.sh"
echo -e "  View logs:     tail -f logs/api.log"
echo -e "  Test API:     curl http://localhost:8000/health"
echo ""