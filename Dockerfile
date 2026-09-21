# Step 1: Use official Playwright environment with Python
FROM mcr.microsoft.com/playwright/python:v1.45.0-jammy

# Step 2: Set working directory inside container
WORKDIR /app

# Step 3: Copy dependency list and install Python packages with higher network timeout
COPY requirements.txt .
RUN pip install --default-timeout=1000 --retries 10 -r requirements.txt

# Step 4: Install necessary Chromium browser binaries inside container
RUN playwright install chromium
RUN playwright install-deps chromium

# Step 5: Copy entire project files
COPY . .

# Step 6: Expose port 8000 for FastAPI & WebSockets
EXPOSE 8000

# Step 7: Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
