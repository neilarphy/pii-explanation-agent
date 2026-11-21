FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee
WORKDIR /service
COPY backend/detectors/requirements.lock /service/requirements.lock
# The base digest pins Python, torch, CUDA and all preinstalled binary libraries.
RUN python -m pip install --no-cache-dir --no-deps -r requirements.lock && python -m pip check
COPY backend/__init__.py /service/backend/__init__.py
COPY backend/detectors/__init__.py /service/backend/detectors/__init__.py
COPY backend/detectors/app /service/backend/detectors/app
COPY backend/detectors/config /service/backend/detectors/config
COPY backend/detectors/scripts /service/backend/detectors/scripts
COPY backend/detectors/tests /service/backend/detectors/tests
COPY backend/detectors/pytest.ini /service/backend/detectors/pytest.ini
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false \
    HF_HOME=/tmp/empty-hf-cache MODEL_PATH=/models \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.detectors.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log", "--log-level", "critical"]
