# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# COPY pyproject.toml first to install dependencies
COPY pyproject.toml README.md ./
# Create a dummy src directory so setuptools doesn't complain during egg_info
RUN mkdir src
# Install only dependencies. 
# We use --no-root is not available in pip, so we install the current dir '.' 
# but setuptools needs the src folder defined in pyproject.toml
RUN pip install --no-cache-dir . --target=/deps

# Stage 2: Runtime image
FROM python:3.11-slim AS runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    GIT_PYTHON_REFRESH=quiet

# Copy only dependencies from builder
COPY --from=builder /deps /usr/local/lib/python3.11/site-packages
# Copy project structure
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY pyproject.toml README.md ./

CMD ["python", "scripts/hello_train.py"]
