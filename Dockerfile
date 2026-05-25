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

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_LINK_MODE=copy

WORKDIR /workspace
