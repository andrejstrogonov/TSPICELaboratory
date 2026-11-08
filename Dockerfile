FROM ubuntu:latest
LABEL authors="strog"

ENTRYPOINT ["top", "-b"]