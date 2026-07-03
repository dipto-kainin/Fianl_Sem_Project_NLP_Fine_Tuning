#!/bin/bash

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}==============================================${NC}"
echo -e "${BLUE}   Starting AI Knowledge Distillation Platform ${NC}"
echo -e "${BLUE}==============================================${NC}"

# 1. Check if Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${YELLOW}Docker daemon is not running! Trying to start it...${NC}"
    sudo systemctl start docker
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}Error: Failed to start Docker daemon. Please start Docker manually.${NC}"
        exit 1
    fi
fi

# 2. Check if services are already running
PID_FILE="./.services.pid"
if [ -f "$PID_FILE" ]; then
    echo -e "${YELLOW}Warning: $PID_FILE exists. Services might already be running.${NC}"
    echo -e "${YELLOW}Run ./stop.sh first if you want to restart them.${NC}"
    exit 1
fi

# 3. Spin up Docker containers (Postgres, Redis, Qdrant)
echo -e "${BLUE}[1/6] Starting Docker containers (Postgres, Redis, Qdrant)...${NC}"
cd backend
docker compose up -d
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to start Docker containers.${NC}"
    exit 1
fi

# 4. Wait for core services to become healthy
echo -e "${BLUE}[2/6] Waiting for core services to become healthy...${NC}"
MAX_RETRIES=30
RETRIES=0
while true; do
    # Get count of healthy containers for this compose project
    HEALTHY_COUNT=$(docker compose ps --format json | grep -c '"Health":"healthy"')
    if [ "$HEALTHY_COUNT" -ge 3 ]; then
        echo -e "${GREEN}All core services (PostgreSQL, Redis, Qdrant) are healthy!${NC}"
        break
    fi
    RETRIES=$((RETRIES+1))
    if [ $RETRIES -ge $MAX_RETRIES ]; then
        echo -e "${YELLOW}Warning: Services are taking longer than expected to report healthy.${NC}"
        echo -e "${YELLOW}Proceeding with startup...${NC}"
        break
    fi
    sleep 1
done

# 5. Run Database Migrations
echo -e "${BLUE}[3/6] Running database migrations...${NC}"
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Python virtual environment (.venv) not found. Creating one...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

alembic upgrade head
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Database migrations failed.${NC}"
    deactivate
    exit 1
fi

# 6. Start Backend FastAPI Application
echo -e "${BLUE}[4/6] Starting Backend FastAPI (Uvicorn)...${NC}"
# Port check
if lsof -t -i:8000 >/dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Port 8000 is already in use! Attempting to free it...${NC}"
    kill -9 $(lsof -t -i:8000) 2>/dev/null
    sleep 1
fi
setsid uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > uvicorn.log 2>&1 &
BACKEND_PID=$!
disown $BACKEND_PID
echo -e "${GREEN}Backend started (PID: $BACKEND_PID, logs: backend/uvicorn.log)${NC}"

# 7. Start Celery Worker
echo -e "${BLUE}[5/6] Starting Celery Worker...${NC}"
setsid celery -A app.workers.celery_app worker --loglevel=info -Q default,documents,teacher,training > celery.log 2>&1 &
CELERY_PID=$!
disown $CELERY_PID
echo -e "${GREEN}Celery worker started (PID: $CELERY_PID, logs: backend/celery.log)${NC}"

deactivate

# 8. Start Frontend React/Next.js Application
echo -e "${BLUE}[6/6] Starting Frontend Application (Next.js)...${NC}"
cd ../frontend
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Frontend node_modules not found. Installing dependencies...${NC}"
    npm install
fi

# Port check
if lsof -t -i:5173 >/dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Port 5173 is already in use! Attempting to free it...${NC}"
    kill -9 $(lsof -t -i:5173) 2>/dev/null
    sleep 1
fi
setsid ./node_modules/.bin/next dev -p 5173 > next.log 2>&1 &
FRONTEND_PID=$!
disown $FRONTEND_PID
echo -e "${GREEN}Frontend started (PID: $FRONTEND_PID, logs: frontend/next.log)${NC}"

# Save PIDs
cd ..
echo "BACKEND_PID=$BACKEND_PID" > "$PID_FILE"
echo "CELERY_PID=$CELERY_PID" >> "$PID_FILE"
echo "FRONTEND_PID=$FRONTEND_PID" >> "$PID_FILE"

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}   All services started successfully!          ${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "   - Backend API:      http://localhost:8000"
echo -e "   - API Docs:         http://localhost:8000/docs"
echo -e "   - Frontend App:     http://localhost:5173"
echo -e "   - Redis:            localhost:6380"
echo -e "   - Qdrant Dashboard: http://localhost:6333/dashboard"
echo -e "   - Postgres:         localhost:5432"
echo -e "${GREEN}================================================${NC}"
echo -e "To stop all services, run: ./stop.sh"
