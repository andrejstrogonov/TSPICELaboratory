# Обновлено до CUDA 12.8 для совместимости с torch==2.7.1+cu128
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:/root/.cargo/bin:$PATH"
ENV PYTHONNOUSERSITE="1"
# Начальный пароль для Neo4j (формат: пользователь/пароль)
ENV NEO4J_AUTH=neo4j/tspice_secret_password

# 1. Установка системных пакетов, утилит для репозиториев и Java 17 (нужна для Neo4j)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    gnupg \
    ca-certificates \
    build-essential \
    git \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# 2. Добавление официального репозитория и установка Neo4j
RUN wget -O - https://neo4j.com | gpg --dearmor -o /etc/apt/keyrings/neotechnology.gpg \
    && echo 'deb [signed-by=/etc/apt/keyrings/neotechnology.gpg] https://neo4j.com stable latest' > /etc/apt/sources.list.d/neo4j.list \
    && apt-get update \
    && apt-get install -y neo4j \
    && rm -rf /var/lib/apt/lists/*

# Разрешаем Neo4j принимать подключения не только с localhost (полезно для отладки)
RUN sed -i 's/#server.default_listen_address=0.0.0.0/server.default_listen_address=0.0.0.0/' /etc/neo4j/neo4j.conf

# 3. Установка Miniconda (используем локальный файл)
COPY Miniconda3-latest-Linux-x86_64.sh /tmp/miniconda.sh
RUN bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# 4. Создание окружения напрямую из environment.yml (без requirements.txt)
WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml \
    && conda clean --all --yes

# 5. Установка Rust (используем приложенный скрипт rustup-init.sh)
COPY rustup-init.sh /tmp/rustup-init.sh
RUN sh /tmp/rustup-init.sh -y \
    && rm /tmp/rustup-init.sh

# 6. Копирование исходного кода приложения
COPY . .

# 7. Создание скрипта для одновременного запуска Neo4j и Django
RUN printf '#!/bin/bash\n\
set -e\n\
echo "Starting Neo4j..."\n\
neo4j start\n\
\n\
# Ожидание готовности порта Bolt (7687)\n\
until nc -z localhost 7687 2>/dev/null || curl -s http://localhost:7474 >/dev/null; do\n\
  echo "Waiting for Neo4j to accept connections..."\n\
  sleep 2\n\
done\n\
\n\
echo "Starting Django server..."\n\
exec conda run --no-capture-output -n tspice-laboratory-e2e python manage.py runserver 0.0.0.0:8000\n' > run_app.sh \
    && chmod +x run_app.sh

# Открываем порты: 8000 (Django), 7474 (Neo4j Browser HTTP), 7687 (Neo4j Bolt)
EXPOSE 8000 7474 7687

CMD ["./run_app.sh"]
