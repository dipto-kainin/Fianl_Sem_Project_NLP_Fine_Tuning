#!/bin/bash

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   Stopping AI Knowledge Distillation Platform   ${NC}"
echo -e "${BLUE}================================================${NC}"

# 1. Stop background processes using PID file
PID_FILE="./.services.pid"
if [ -f "$PID_FILE" ]; then
    source "$PID_FILE"
    
    if [ ! -z "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "Stopping Backend (PID $BACKEND_PID)..."
        kill "$BACKEND_PID" 2>/dev/null
    fi
    
    if [ ! -z "$CELERY_PID" ] && kill -0 "$CELERY_PID" 2>/dev/null; then
        echo -e "Stopping Celery worker (PID $CELERY_PID)..."
        kill "$CELERY_PID" 2>/dev/null
    fi
    
    if [ ! -z "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo -e "Stopping Frontend (PID $FRONTEND_PID)..."
        kill "$FRONTEND_PID" 2>/dev/null
    fi
    
    rm "$PID_FILE"
else
    echo -e "${YELLOW}No active .services.pid file found. Cleaning up by ports/process name...${NC}"
fi

# 2. Hard cleanup of orphaned processes
echo -e "${BLUE}Ensuring no process is left running on ports 8000 and 5173...${NC}"
BACKEND_PORT_PID=$(lsof -t -i:8000)
if [ ! -z "$BACKEND_PORT_PID" ]; then
    echo -e "Cleaning up process on port 8000 (PID $BACKEND_PORT_PID)..."
    kill -9 $BACKEND_PORT_PID 2>/dev/null
fi

FRONTEND_PORT_PID=$(lsof -t -i:5173)
if [ ! -z "$FRONTEND_PORT_PID" ]; then
    echo -e "Cleaning up process on port 5173 (PID $FRONTEND_PORT_PID)..."
    kill -9 $FRONTEND_PORT_PID 2>/dev/null
fi

CELERY_PIDS=$(pgrep -f "celery -A app.workers.celery_app")
if [ ! -z "$CELERY_PIDS" ]; then
    echo -e "Cleaning up Celery workers (PIDs $CELERY_PIDS)..."
    kill -9 $CELERY_PIDS 2>/dev/null
fi

# 3. Stop Docker containers
echo -e "${BLUE}Stopping Docker containers...${NC}"
cd backend
docker compose down
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Docker containers stopped and removed successfully.${NC}"
else
    echo -e "${RED}Error: Failed to stop Docker containers.${NC}"
fi

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}   All services stopped successfully!           ${NC}"
echo -e "${GREEN}================================================${NC}"
