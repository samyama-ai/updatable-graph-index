# amd64 image (mini is arm64 → emulated) with the ANN churn-harness deps prebuilt, so parallel
# sweep runs skip repeated pip installs. Build once on mini:
#   docker build --platform linux/amd64 -t uidx:latest .
FROM --platform=linux/amd64 python:3.11-slim
RUN pip install --no-cache-dir diskannpy numpy h5py
WORKDIR /work
