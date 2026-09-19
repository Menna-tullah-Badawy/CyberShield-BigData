# ==============================================================================
# CyberShield-BigData Production Docker Container
# Multi-stage build with Java 17, Apache Spark, PySpark, PyTorch & Transformers
# ==============================================================================
FROM python:3.10-slim-bullseye

# منع إنشاء ملفات .pyc وإلغاء تخزين المخرجات في الذاكرة المؤقتة
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    SPARK_HOME=/opt/spark \
    PATH="/usr/lib/jvm/java-17-openjdk-amd64/bin:/opt/spark/bin:${PATH}"

# تثبيت OpenJDK والأدوات الأساسية
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk-headless \
    curl \
    gcc \
    g++ \
    libpcap-dev \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# تحديد مجلد العمل
WORKDIR /app

# نسخ ملف الاعتماديات وتثبيت حزم بايثون
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# نسخ ملفات المشروع بالكامل
COPY . .

# إنشاء مجلدات التخزين المؤقتة والتقارير وضبط الصلاحيات
RUN mkdir -p data/raw data/processed data/features data/checkpoints models reports logs \
    && chmod -R 777 data models reports logs

# المنفذ 8000 لخدمة FastAPI والمنفذ 8501 للوحة تحكم Streamlit
EXPOSE 8000 8501

# أمر التشغيل الافتراضي (FastAPI Serving)
CMD ["uvicorn", "serving.api.main:app", "--host", "0.0.0.0", "--port", "8000"]