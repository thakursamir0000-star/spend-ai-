# Build and run:
#   docker build -t spend-ai . && docker run -p 8501:8501 -e GROQ_API_KEY=gsk_... spend-ai
FROM python:3.12-slim

# Run as a non-root user with a writable home so the categorization cache can be saved.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
