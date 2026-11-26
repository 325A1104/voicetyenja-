FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg p7zip-full wget curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/voicevox /tmp/vv && \
    ASSET_URLS=$(curl -s https://api.github.com/repos/VOICEVOX/voicevox_engine/releases/latest \
    | grep "browser_download_url" | grep "linux-cpu-x64" | grep "\.7z\." | cut -d '"' -f 4 | sort) && \
    for u in $ASSET_URLS; do wget -q "$u" -P /tmp/vv/; done && \
    FIRST_PART=$(ls /tmp/vv | sort | head -n 1) && \
    7z x -y "/tmp/vv/$FIRST_PART" -o/opt/voicevox/ && \
    mv /opt/voicevox/*linux-cpu-x64*/* /opt/voicevox/ && \
    chmod +x /opt/voicevox/run && \
    rm -rf /tmp/vv

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV COQUI_TOS_AGREED=1
ENV HOME=/app
EXPOSE 7860

CMD ["/bin/bash", "-c", "/opt/voicevox/run --port 50021 & sleep 8 && streamlit run app.py --server.port 7860 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false"]
