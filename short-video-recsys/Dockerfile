FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY models.py serve.py ./
ENV OMP_NUM_THREADS=1
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
