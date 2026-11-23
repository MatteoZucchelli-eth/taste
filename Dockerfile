# Reproducible GPU environment for the taste project.
# Base: official PyTorch image (torch 2.15.0, CUDA 13.2, cuDNN 9) — tested on RTX 5070 Ti (Blackwell, driver 580).
FROM pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib

COPY requirements.txt /tmp/requirements.txt
# The base image installs torch into the system Python (PEP 668 "externally managed"); overriding that is fine in a container.
# The final check fails the build if pip replaced the base image's CUDA torch.
RUN pip install --break-system-packages -r /tmp/requirements.txt \
 && python -c "import torch; assert torch.__version__.startswith('2.15.0') and torch.version.cuda == '13.2', torch.__version__"

WORKDIR /workspace
CMD ["bash"]
