#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
#  clean.sh  —  AI Knowledge Distillation Platform  ·  DB Cleaner
# ──────────────────────────────────────────────────────────────────

set -uo pipefail

# ── Colours ──────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

# ── Config ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PG_CONTAINER="kdp_postgres"
PG_USER="kdp_user"
PG_DB="knowledge_distillation"
QDRANT_URL="http://localhost:6333"
QDRANT_COLLECTION="knowledge_base"
STORAGE_DIR="${SCRIPT_DIR}/backend/storage"
DOCKER_DIR="${SCRIPT_DIR}/backend"

# Track if we brought Docker up ourselves so we can tear it down after
_DOCKER_STARTED_BY_US=0

# ── Docker lifecycle ──────────────────────────────────────────────
ensure_docker_up() {
  if docker inspect "$PG_CONTAINER" &>/dev/null 2>&1; then
    return 0  # already running
  fi
  echo ""
  echo -e "  ${YELLOW}Docker containers are not running — starting them temporarily...${RESET}"
  docker compose -f "${DOCKER_DIR}/docker-compose.yml" up -d --quiet-pull > /dev/null 2>&1
  # Wait for Postgres to be ready
  local retries=20
  while ! docker exec "$PG_CONTAINER" pg_isready -U "$PG_USER" -d "$PG_DB" &>/dev/null 2>&1; do
    retries=$((retries - 1))
    if [[ $retries -le 0 ]]; then
      echo -e "  ${RED}✗  Timed out waiting for Postgres to be ready.${RESET}"
      exit 1
    fi
    sleep 1
  done
  ok "Docker containers started"
  _DOCKER_STARTED_BY_US=1
}

docker_teardown() {
  if [[ $_DOCKER_STARTED_BY_US -eq 1 ]]; then
    info "Stopping Docker containers (we started them for cleanup)..."
    docker compose -f "${DOCKER_DIR}/docker-compose.yml" down > /dev/null 2>&1
    ok "Docker containers stopped"
  fi
}

# ── Helpers ──────────────────────────────────────────────────────
pg() { docker exec -i "$PG_CONTAINER" psql -U "$PG_USER" -d "$PG_DB" -c "$1"; }

confirm() {
  echo -e "${YELLOW}⚠  $1${RESET}"
  read -rp "   Type YES to confirm: " ans
  [[ "$ans" == "YES" ]]
}

header() {
  echo ""
  echo -e "${CYAN}${BOLD}──────────────────────────────────────────${RESET}"
  echo -e "${CYAN}${BOLD}  $1${RESET}"
  echo -e "${CYAN}${BOLD}──────────────────────────────────────────${RESET}"
}

ok()   { echo -e "  ${GREEN}✓${RESET}  $1"; }
info() { echo -e "  ${DIM}·${RESET}  $1"; }
skip() { echo -e "  ${DIM}–${RESET}  $1 (skipped)"; }

# ── Individual actions ────────────────────────────────────────────

clear_documents() {
  info "Truncating documents + chunks + teacher_outputs..."
  pg "TRUNCATE TABLE teacher_outputs, chunks, documents RESTART IDENTITY CASCADE;" > /dev/null
  ok "Documents, chunks, teacher outputs cleared"
}

clear_datasets() {
  info "Truncating datasets + dataset_samples..."
  pg "TRUNCATE TABLE dataset_samples, datasets RESTART IDENTITY CASCADE;" > /dev/null
  ok "Datasets cleared from DB"

  info "Removing dataset files from storage..."
  rm -rf "${STORAGE_DIR}/datasets/"*
  ok "storage/datasets/ wiped"
}

clear_training() {
  info "Truncating training_runs + model_versions..."
  pg "TRUNCATE TABLE training_runs, model_versions RESTART IDENTITY CASCADE;" > /dev/null
  ok "Training runs + model versions cleared from DB"

  info "Removing model files from storage..."
  rm -rf "${STORAGE_DIR}/models/"*
  ok "storage/models/ wiped"
}

clear_uploads() {
  info "Removing uploaded files from storage..."
  rm -rf "${STORAGE_DIR}/uploads/"*
  ok "storage/uploads/ wiped"
}

clear_qdrant() {
  info "Deleting Qdrant collection '${QDRANT_COLLECTION}'..."
  local res
  res=$(curl -s -X DELETE "${QDRANT_URL}/collections/${QDRANT_COLLECTION}")
  if echo "$res" | grep -q '"result":true'; then
    ok "Qdrant collection deleted"
  else
    echo -e "  ${YELLOW}⚠${RESET}  Qdrant: ${res}"
  fi
}

clear_redis() {
  info "Flushing Redis queues..."
  if docker exec -i kdp_redis redis-cli flushall &>/dev/null; then
    ok "Redis queues flushed"
  else
    echo -e "  ${YELLOW}⚠${RESET}  Failed to flush Redis"
  fi
}

# ── Menu options ──────────────────────────────────────────────────

do_full_reset() {
  header "1 · FULL RESET"
  confirm "This will wipe ALL data — DB tables, files, and Qdrant." || { echo "Aborted."; exit 0; }
  echo ""
  ensure_docker_up
  clear_documents
  clear_datasets
  clear_training
  clear_uploads
  clear_qdrant
  clear_redis
  docker_teardown
  echo ""
  echo -e "${GREEN}${BOLD}  Full reset complete. Fresh slate.${RESET}"
}

do_documents_and_datasets() {
  header "2 · DOCUMENTS + DATASETS RESET"
  confirm "This will clear documents, chunks, teacher outputs, datasets (DB + files), and uploads." || { echo "Aborted."; exit 0; }
  echo ""
  ensure_docker_up
  clear_documents
  clear_datasets
  clear_uploads
  clear_qdrant
  clear_redis
  docker_teardown
  echo ""
  echo -e "${GREEN}${BOLD}  Documents & datasets cleared.${RESET}"
}

do_datasets_only() {
  header "3 · DATASETS ONLY"
  confirm "This will clear datasets + dataset_samples (DB + files). Documents stay intact." || { echo "Aborted."; exit 0; }
  echo ""
  ensure_docker_up
  clear_datasets
  docker_teardown
  echo ""
  echo -e "${GREEN}${BOLD}  Datasets cleared.${RESET}"
}

do_models_only() {
  header "4 · MODELS + TRAINING RUNS ONLY"
  confirm "This will clear training_runs + model_versions (DB + model files). Everything else stays." || { echo "Aborted."; exit 0; }
  echo ""
  ensure_docker_up
  clear_training
  docker_teardown
  echo ""
  echo -e "${GREEN}${BOLD}  Models & training runs cleared.${RESET}"
}

do_qdrant_only() {
  header "5 · QDRANT VECTOR STORE ONLY"
  confirm "This will delete and recreate the Qdrant collection. DB rows stay intact." || { echo "Aborted."; exit 0; }
  echo ""
  ensure_docker_up
  clear_qdrant
  docker_teardown
  echo ""
  echo -e "${GREEN}${BOLD}  Qdrant cleared.${RESET}"
}

do_custom() {
  header "6 · CUSTOM — pick what to clear"
  echo ""
  declare -A chosen=()

  pick() {
    read -rp "  $1? [y/N] " ans
    [[ "${ans,,}" == "y" ]]
  }

  pick "Documents, chunks & teacher outputs (DB)"      && chosen[docs]=1
  pick "Uploaded files (storage/uploads/)"             && chosen[uploads]=1
  pick "Datasets (DB rows + storage/datasets/)"        && chosen[datasets]=1
  pick "Training runs & model versions (DB)"           && chosen[training]=1
  pick "Model files (storage/models/)"                 && chosen[models]=1
  pick "Qdrant vector store"                           && chosen[qdrant]=1

  if [[ ${#chosen[@]} -eq 0 ]]; then
    echo "Nothing selected. Aborted."; exit 0
  fi

  echo ""
  confirm "Proceed with selected cleanup?" || { echo "Aborted."; exit 0; }

  # Only start Docker if we need DB access
  local needs_docker=0
  [[ -n "${chosen[docs]+x}" || -n "${chosen[datasets]+x}" || -n "${chosen[training]+x}" || -n "${chosen[qdrant]+x}" ]] && needs_docker=1
  [[ $needs_docker -eq 1 ]] && ensure_docker_up

  echo ""

  [[ -n "${chosen[docs]+x}" ]]     && pg "TRUNCATE TABLE teacher_outputs, chunks, documents RESTART IDENTITY CASCADE;" > /dev/null && ok "Documents cleared"
  [[ -n "${chosen[uploads]+x}" ]]  && rm -rf "${STORAGE_DIR}/uploads/"* && ok "Uploads cleared"
  [[ -n "${chosen[datasets]+x}" ]] && pg "TRUNCATE TABLE dataset_samples, datasets RESTART IDENTITY CASCADE;" > /dev/null && rm -rf "${STORAGE_DIR}/datasets/"* && ok "Datasets cleared"
  [[ -n "${chosen[training]+x}" ]] && pg "TRUNCATE TABLE training_runs, model_versions RESTART IDENTITY CASCADE;" > /dev/null && ok "Training runs cleared"
  [[ -n "${chosen[models]+x}" ]]   && rm -rf "${STORAGE_DIR}/models/"* && ok "Model files cleared"
  [[ -n "${chosen[qdrant]+x}" ]]   && clear_qdrant

  docker_teardown
  echo ""
  echo -e "${GREEN}${BOLD}  Custom cleanup complete.${RESET}"
}

# ── Main menu ─────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}  🧹  KDP Database Cleaner${RESET}"
echo -e "${DIM}  ──────────────────────────────────────────${RESET}"
echo ""
echo -e "  ${BOLD}1${RESET}  Full reset            ${DIM}(all tables + all files + Qdrant)${RESET}"
echo -e "  ${BOLD}2${RESET}  Documents & datasets  ${DIM}(docs, chunks, teacher, datasets, uploads, Qdrant)${RESET}"
echo -e "  ${BOLD}3${RESET}  Datasets only         ${DIM}(dataset tables + files only)${RESET}"
echo -e "  ${BOLD}4${RESET}  Models & training     ${DIM}(training_runs, model_versions, model files)${RESET}"
echo -e "  ${BOLD}5${RESET}  Qdrant only           ${DIM}(vector store only)${RESET}"
echo -e "  ${BOLD}6${RESET}  Custom                ${DIM}(choose what to clear interactively)${RESET}"
echo -e "  ${BOLD}q${RESET}  Quit"
echo ""
read -rp "  Choose [1-6 / q]: " choice
echo ""

case "$choice" in
  1) do_full_reset ;;
  2) do_documents_and_datasets ;;
  3) do_datasets_only ;;
  4) do_models_only ;;
  5) do_qdrant_only ;;
  6) do_custom ;;
  q|Q) echo "Bye."; exit 0 ;;
  *) echo -e "${RED}Invalid option.${RESET}"; exit 1 ;;
esac

echo ""
