FROM python:3.12-slim-bookworm

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        ca-certificates curl fonts-ipafont-gothic gcc git locales sudo tmux tzdata vim zsh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "ja_JP UTF-8" > /etc/locale.gen && \
    locale-gen ja_JP.UTF-8
ENV LANG=ja_JP.UTF-8
ENV LC_ALL=ja_JP.UTF-8
ENV TZ=Asia/Tokyo

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_LINK_MODE=copy

WORKDIR /workspace
