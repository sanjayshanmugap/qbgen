# Stage 1: Build the frontend
FROM node:20 AS frontend-builder

# Set working directory for frontend
WORKDIR /app/frontend

# Copy package.json and package-lock.json, then install dependencies
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install

# Copy the rest of the frontend files and build
COPY frontend/ ./
RUN npm run build

# Stage 2: Set up the backend and serve both backend and frontend
FROM python:3.11 AS backend

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory for backend
WORKDIR /app

# Copy backend dependencies and install them
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy backend source code
COPY backend/ ./

# Copy the frontend build output into the backend's static directory
COPY --from=frontend-builder /app/frontend/dist /app/static

# Expose Flask development server's default port
EXPOSE 8080

# Start the Flask development server
CMD ["python", "app.py"]