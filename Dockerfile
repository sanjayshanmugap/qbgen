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

# Copy frontend source code (excluding node_modules to avoid conflicts)
COPY qbgen-landing/app ./app
COPY qbgen-landing/components ./components
COPY qbgen-landing/lib ./lib
COPY qbgen-landing/hooks ./hooks
COPY qbgen-landing/public ./public
COPY qbgen-landing/styles ./styles

# Copy configuration files (use individual files to avoid wildcard issues)
COPY qbgen-landing/next.config.mjs ./
COPY qbgen-landing/tsconfig.json ./
COPY qbgen-landing/tailwind.config.js ./
COPY qbgen-landing/postcss.config.mjs ./
COPY qbgen-landing/components.json ./

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

# Copy model download script and download the model
COPY download_model.py .
RUN python download_model.py

# Copy backend source code
COPY backend/ ./backend/

# Copy the built Next.js static export from the frontend-builder stage
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