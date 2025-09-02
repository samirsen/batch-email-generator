# Alpine-based Dockerfile (bypasses Debian repository issues)
FROM python:3.11-alpine as builder

# Set working directory
WORKDIR /app

# Install build dependencies for Alpine
RUN apk add --no-cache --virtual .build-deps \
    gcc \
    g++ \
    musl-dev \
    postgresql-dev \
    libffi-dev \
    openssl-dev \
    && apk add --no-cache \
    curl \
    ca-certificates

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-alpine

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apk add --no-cache \
    postgresql-libs \
    curl \
    ca-certificates

# Create non-root user for security
RUN addgroup -g 1001 -S appuser && \
    adduser -S appuser -G appuser

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/temp /app/logs

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Direct command without entrypoint for simplicity
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
