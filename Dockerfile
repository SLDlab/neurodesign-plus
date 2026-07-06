FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir .

CMD ["python", "-c", "from neurodesign import Experiment, Design, Optimisation; print('neurodesign-plus container ready')"]
