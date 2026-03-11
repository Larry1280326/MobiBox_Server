#!/bin/bash
# MobiBox Backend - Restart All Services
# This script restarts all MobiBox backend services

set -e

# Colors for output
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}Restarting MobiBox Backend services...${NC}"
echo ""

# Stop services
"$SCRIPT_DIR/stop_services.sh"

echo ""
echo "Waiting 3 seconds..."
sleep 3

# Start services
"$SCRIPT_DIR/start_services.sh"