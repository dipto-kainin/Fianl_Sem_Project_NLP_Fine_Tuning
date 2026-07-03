#!/bin/bash

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}==============================================${NC}"
echo -e "${BLUE}   AI Knowledge Distillation Platform - Logs   ${NC}"
echo -e "${BLUE}==============================================${NC}"
echo -e "Choose which log to view:"
echo -e "1) Backend API (Uvicorn)"
echo -e "2) Celery Worker"
echo -e "3) Frontend (Next.js)"
echo -e "4) Docker Containers (Compose logs)"
echo -e "5) Stream all log files together"
echo -e "q) Exit"
echo -e "${BLUE}==============================================${NC}"

read -p "Enter your choice (1-5 or q): " choice

case $choice in
    1)
        echo -e "${GREEN}Streaming Backend API (Uvicorn) logs... (Press Ctrl+C to stop)${NC}"
        tail -f backend/uvicorn.log
        ;;
    2)
        echo -e "${GREEN}Streaming Celery Worker logs... (Press Ctrl+C to stop)${NC}"
        tail -f backend/celery.log
        ;;
    3)
        echo -e "${GREEN}Streaming Frontend (Next.js) logs... (Press Ctrl+C to stop)${NC}"
        tail -f frontend/next.log
        ;;
    4)
        echo -e "${GREEN}Streaming Docker Compose logs... (Press Ctrl+C to stop)${NC}"
        cd backend && docker compose logs -f
        ;;
    5)
        echo -e "${GREEN}Streaming all application logs... (Press Ctrl+C to stop)${NC}"
        tail -f backend/uvicorn.log backend/celery.log frontend/next.log
        ;;
    q|Q)
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice.${NC}"
        ;;
esac
