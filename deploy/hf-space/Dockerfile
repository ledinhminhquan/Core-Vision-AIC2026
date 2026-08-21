# Round-42 · Web luyện tập 24/7 (Hugging Face Space · Docker · CPU Basic 16GB)
# Space secrets cần khai báo (Settings → Variables and secrets):
#   GITHUB_TOKEN  (secret, fine-grained, Contents: Read — clone repo private lúc BUILD)
#   HF_TOKEN      (secret, Read — tải 2 dataset private lúc KHỞI ĐỘNG)
#   TEAM_PASS     (secret, mật khẩu đội)  ·  TEAM_USER (variable, vd aic2026-222)
#   ARTIFACTS_REPO (variable, vd <user>/aic26-artifacts)
#   THUMBS_REPO    (variable, vd <user>/aic26-thumbs)
#   GEMINI_API_KEY (secret, tùy chọn — bật Gemini dịch/enhance cho luyện tập)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Torch CPU trước (wheel riêng), rồi phần còn lại.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" huggingface_hub hf_transfer \
    transformers safetensors sentencepiece protobuf \
    faiss-cpu rank-bm25 pandas pyarrow numpy pillow pydantic pydantic-settings \
    google-genai deep-translator pyyaml

# Clone repo PRIVATE bằng build-secret (không lộ token vào layer).
RUN --mount=type=secret,id=GITHUB_TOKEN,mode=0444,required=true \
    git clone --depth 1 \
    "https://$(cat /run/secrets/GITHUB_TOKEN)@github.com/ledinhminhquan/Core-Vision_Perfect_V1.git" \
    /app/repo \
    && pip install --no-cache-dir --no-deps -e /app/repo

COPY app.py /app/app.py
WORKDIR /app
# User không-root theo khuyến nghị HF; cache HF về thư mục ghi được.
RUN useradd -m appuser && mkdir -p /home/appuser/hf && chown -R appuser /app /home/appuser
USER appuser
ENV HF_HOME=/home/appuser/hf HF_HUB_ENABLE_HF_TRANSFER=1

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
