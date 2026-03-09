FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download modeller ved build-tid (caches i image → hurtig cold start)
RUN python -c "\
from qwen_tts import Qwen3TTSModel; \
print('Downloader Base model...'); \
Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base', device_map='cpu', dtype='auto'); \
print('Downloader VoiceDesign model...'); \
Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign', device_map='cpu', dtype='auto'); \
print('Modeller downloadet!')"

# Handler
COPY handler.py .

CMD ["python", "handler.py"]
