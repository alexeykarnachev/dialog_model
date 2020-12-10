#!/bin/sh

PORT=$1
WORKERS=$2
LOGS_DIR=$3
EXPERIMENT_DIR=$4
CHECKPOINT_NAME=$5
DEVICE=$6

TIMEOUT=600
WORKER_CLASS="uvicorn.workers.UvicornWorker"
MODULE_PATH="dialog_model.api.app"
ACCESS_LOGFILE=$LOGS_DIR"/gunicorn_access.log"
ERROR_LOGFILE=$LOGS_DIR"/gunicorn_error.log"
LOG_LEVEL="DEBUG"

mkdir -p "${LOGS_DIR}"

exec gunicorn ${MODULE_PATH}":prepare(experiment_dir='${EXPERIMENT_DIR}', checkpoint_name='${CHECKPOINT_NAME}', device='${DEVICE}', logs_dir='${LOGS_DIR}')" \
-b :"${PORT}" \
--timeout "${TIMEOUT}" \
-k "${WORKER_CLASS}" \
--workers "${WORKERS}" \
--access-logfile "${ACCESS_LOGFILE}" \
--error-logfile "${ERROR_LOGFILE}" \
--log-level "${LOG_LEVEL}"
