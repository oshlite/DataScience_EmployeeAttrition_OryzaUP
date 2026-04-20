FROM python:3.11-slim

LABEL Name=dspt1oryzaemployeeattrition Version=0.0.1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "web_app:app"]
