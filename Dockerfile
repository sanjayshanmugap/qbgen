# Multi-stage build for Next.js frontend and Python backend
FROM node:18-alpine AS frontend-builder

# Set working directory for frontend
WORKDIR /app/frontend

# Copy package files first for better caching
COPY qbgen-landing/package*.json ./
COPY qbgen-landing/pnpm-lock.yaml ./

# Install pnpm
RUN npm install -g pnpm

# Install dependencies
RUN pnpm install --frozen-lockfile

# Copy frontend source code
COPY qbgen-landing/ ./

# Build the Next.js application
RUN pnpm build

# Production stage with Python backend
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy backend requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copy backend source code
COPY backend/ ./backend/

# Copy the built Next.js frontend static files
COPY --from=frontend-builder /app/frontend/out/ ./static/

# Create a simple startup script
RUN echo '#!/bin/bash\n\
cd /app/backend\n\
python app.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose port
EXPOSE 8080

# Start the application
CMD ["/app/start.sh"]