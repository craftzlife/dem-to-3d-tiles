FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.4

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    cmake \
    g++ \
    libdeflate-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/

RUN pip3 install --no-cache-dir pytest pytest-cov

ENTRYPOINT ["python3", "-m", "src.main"]
