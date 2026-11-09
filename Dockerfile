FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Установка необходимых пакетов
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Установка Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

# Добавление conda в PATH
ENV PATH="/opt/conda/bin:$PATH"
ENV PYTHONNOUSERSITE="1"

# Создание окружения с указанными зависимостями
RUN conda create -n tspice-laboratory-e2e \
    --channel conda-forge \
    --channel nodefaults \
    python=3.12 \
    onnx>=1.16 \
    sqlite \
    pip \
    --yes

# Копирование requirements.txt
COPY requirements.txt .

# Установка pip-зависимостей в созданное окружение
RUN conda run -n tspice-laboratory-e2e pip install -r requirements.txt

# Установка Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Установка дополнительных библиотек для GPU поддержки
RUN apt-get update && apt-get install -y \
    nvidia-driver-470 \
    && rm -rf /var/lib/apt/lists/*

# Создание директории для приложения
RUN mkdir -p /app
WORKDIR /app

# Копирование исходного кода
COPY . .

# Создание скрипта для запуска
RUN echo '#!/bin/bash\n\
conda run -n tspice-laboratory-e2e python manage.py runserver 0.0.0.0:8000' > run_app.sh \
    && chmod +x run_app.sh

CMD ["./run_app.sh"]
