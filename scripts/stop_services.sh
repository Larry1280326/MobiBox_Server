#!/bin/bash
# MobiBox Backend - Stop All Services
# This script stops all running MobiBox backend services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/logs"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  MobiBox Backend - Service Shutdown${NC}"
echo -e "${BLUE}========================================${NC}"

# =========================================
# Stop Celery Beat
# =========================================
echo -e "${YELLOW}Stopping Celery Beat...${NC}"
if pgrep -f "celery.*beat" > /dev/null; then
    pkill -f "celery.*beat" 2>/dev/null || true
    echo -e "${GREEN}✓ Celery Beat stopped${NC}"
else
    echo -e "${YELLOW}Celery Beat was not running${NC}"
fi

# =========================================
# Stop Celery Worker
# =========================================
echo -e "${YELLOW}Stopping Celery Worker...${NC}"
if pgrep -f "celery.*worker" > /dev/null; then
    pkill -f "celery.*worker" 2>/dev/null || true
    echo -e "${GREEN}✓ Celery Worker stopped${NC}"
else
    echo -e "${YELLOW}Celery Worker was not running${NC}"
fi

# =========================================
# Stop FastAPI Server
# =========================================
echo -e "${YELLOW}Stopping FastAPI Server...${NC}"
if pgrep -f "uvicorn src.main:app" > /dev/null; then
    pkill -f "uvicorn src.main:app" 2>/dev/null || true
    echo -e "${GREEN}✓ FastAPI Server stopped${NC}"
else
    echo -e "${YELLOW}FastAPI Server was not running${NC}"
fi

# =========================================
# Stop RabbitMQ (Optional)
# =========================================
echo ""
echo -e "${YELLOW}Do you want to stop RabbitMQ? (y/N)${NC}"
read -t 5 -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Stopping RabbitMQ...${NC}"
    docker stop rabbitmq 2>/dev/null || {
        echo -e "${YELLOW}RabbitMQ container not found or not running${NC}"
    }
    echo -e "${GREEN}✓ RabbitMQ stopped${NC}"
else
    echo -e "${YELLOW}Keeping RabbitMQ running (use 'docker stop rabbitmq' to stop manually)${NC}"
fi

# =========================================
# Clean up PID files
# =========================================
echo -e "${YELLOW}Cleaning up PID files...${NC}"
rm -f "$LOGS_DIR"/*.pid 2>/dev/null || true

# =========================================
# Summary
# =========================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  All services stopped${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}To restart services:${NC}"
echo -e "  ./scripts/start_services.sh"
echo ""