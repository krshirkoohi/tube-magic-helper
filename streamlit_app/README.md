# YouTube SEO Helper – Streamlit Version

This folder contains a lightweight Streamlit web application wrapping the core YouTube SEO Helper CLI functionality.

## Quickstart (local)

```bash
# From project root
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

Open http://localhost:8501 in your browser.

## Running with Docker

```bash
# Build the image
docker build -t tube-magic-helper:latest -f streamlit_app/Dockerfile .

# Run the container
# Ensure .env is provided (mount or pass env vars)
docker run -p 8501:8501 --env-file .env tube-magic-helper:latest
```

## Deployment

The generated image can be deployed to any container-friendly platform (Railway, Render, Fly.io, etc.). Ensure environment variables `YOUTUBE_API_KEY` and `OPENAI_API_KEY` are set.
