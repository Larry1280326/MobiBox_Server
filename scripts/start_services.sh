#!/bin/bash
# MobiBox Backend - Start All Services
# This script starts all required services for MobiBox backend

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

# Create logs directory
mkdir -p "$LOGS_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  MobiBox Backend - Service Startup${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    exit 1
fi

# Activate conda environment
echo -e "${YELLOW}Activating conda environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate Mobibox_backend 2>/dev/null || {
    echo -e "${RED}Error: Failed to activate conda environment 'Mobibox_backend'${NC}"
    echo -e "${YELLOW}Make sure you've created the environment:${NC}"
    echo -e "  conda env create -f environment.yml"
    exit 1
}

# Check for .env file
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${YELLOW}Warning: .env file not found. Copying from .env.example...${NC}"
    if [ -f "$PROJECT_ROOT/.env.example" ]; then
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        echo -e "${YELLOW}Please edit .env with your credentials before running again.${NC}"
    else
        echo -e "${RED}Error: .env.example not found. Please create .env manually.${NC}"
        exit 1
    fi
fi

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for a service to be ready
wait_for_service() {
    local name=$1
    local port=$2
    local max_attempts=30
    local attempt=1

    echo -e "${YELLOW}Waiting for $name to be ready...${NC}"
    while ! check_port $port; do
        if [ $attempt -ge $max_attempts ]; then
            echo -e "${RED}Error: $name failed to start after $max_attempts seconds${NC}"
            return 1
        fi
        sleep 1
        ((attempt++))
    done
    echo -e "${GREEN}✓ $name is ready${NC}"
}

# =========================================
# Step 1: Check/Start RabbitMQ
# =========================================
echo -e "${BLUE}Step 1: RabbitMQ (Message Queue)${NC}"

if docker ps | grep -q rabbitmq; then
    echo -e "${GREEN}✓ RabbitMQ container is already running${NC}"
else
    echo -e "${YELLOW}Starting RabbitMQ container...${NC}"
    docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management 2>/dev/null || {
        echo -e "${YELLOW}Container may already exist, starting it...${NC}"
        docker start rabbitmq 2>/dev/null || {
            echo -e "${RED}Error: Failed to start RabbitMQ${NC}"
            exit 1
        }
    }
    sleep 5  # Wait for RabbitMQ to initialize
fi

# =========================================
# Step 2: Start FastAPI Server
# =========================================
echo -e "${BLUE}Step 2: FastAPI Server${NC}"

if check_port 8000; then
    echo -e "${YELLOW}Port 8000 is already in use. Skipping FastAPI server startup.${NC}"
    echo -e "${YELLOW}To restart, run: pkill -f 'uvicorn src.main:app'${NC}"
else
    echo -e "${YELLOW}Starting FastAPI server...${NC}"
    cd "$PROJECT_ROOT"
    # Logs are handled by Python's RotatingFileHandler in src/logging_config.py
    # Redirect stdout/stderr to /dev/null to avoid nohup.out
    nohup uvicorn src.main:app --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &
    echo $! > "$LOGS_DIR/api.pid"
    wait_for_service "FastAPI" 8000
fi

# =========================================
# Step 3: Start Celery Worker
# =========================================
echo -e "${BLUE}Step 3: Celery Worker${NC}"

# Check if celery worker is running
if pgrep -f "celery.*worker" > /dev/null; then
    echo -e "${YELLOW}Celery worker is already running. Skipping.${NC}"
    echo -e "${YELLOW}To restart, run: pkill -f 'celery.*worker'${NC}"
else
    echo -e "${YELLOW}Starting Celery worker...${NC}"
    cd "$PROJECT_ROOT"
    # Logs are handled by Python's RotatingFileHandler in src/logging_config.py
    # Redirect stdout/stderr to /dev/null to avoid nohup.out
    nohup celery -A src.celery_app.celery_app worker --loglevel=info > /dev/null 2>&1 &
    echo $! > "$LOGS_DIR/celery_worker.pid"
    sleep 3
    echo -e "${GREEN}✓ Celery worker started${NC}"
fi

# =========================================
# Step 4: Start Celery Beat (Scheduler)
# =========================================
echo -e "${BLUE}Step 4: Celery Beat (Scheduler)${NC}"

# Check if celery beat is running
if pgrep -f "celery.*beat" > /dev/null; then
    echo -e "${YELLOW}Celery beat is already running. Skipping.${NC}"
    echo -e "${YELLOW}To restart, run: pkill -f 'celery.*beat'${NC}"
else
    echo -e "${YELLOW}Starting Celery beat...${NC}"
    cd "$PROJECT_ROOT"
    # Logs are handled by Python's RotatingFileHandler in src/logging_config.py
    # Redirect stdout/stderr to /dev/null to avoid nohup.out
    nohup celery -A src.celery_app.celery_app beat --loglevel=info > /dev/null 2>&1 &
    echo $! > "$LOGS_DIR/celery_beat.pid"
    sleep 2
    echo -e "${GREEN}✓ Celery beat started${NC}"
fi

# =========================================
# Summary
# =========================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  All services started successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Service Status:${NC}"
echo -e "  ${GREEN}✓${NC} RabbitMQ:      http://localhost:15672 (guest/guest)"
echo -e "  ${GREEN}✓${NC} FastAPI:        http://localhost:8000"
echo -e "  ${GREEN}✓${NC} Celery Worker:  Running"
echo -e "  ${GREEN}✓${NC} Celery Beat:    Running"
echo ""
echo -e "${BLUE}Logs Location:${NC}"
echo -e "  API Server:     $LOGS_DIR/api.log"
echo -e "  Celery Worker:  $LOGS_DIR/celery_worker.log"
echo -e "  Celery Beat:    $LOGS_DIR/celery_beat.log"
echo ""
echo -e "${BLUE}PID Files:${NC}"
echo -e "  API Server:     $LOGS_DIR/api.pid"
echo -e "  Celery Worker:  $LOGS_DIR/celery_worker.pid"
echo -e "  Celery Beat:    $LOGS_DIR/celery_beat.pid"
echo ""
echo -e "${BLUE}Quick Commands:${NC}"
echo -e "  View API logs:      tail -f $LOGS_DIR/api.log"
echo -e "  View worker logs:   tail -f $LOGS_DIR/celery_worker.log"
echo -e "  Stop all services:  ./scripts/stop_services.sh"
echo -e "  Check health:       curl http://localhost:8000/health"
echo ""