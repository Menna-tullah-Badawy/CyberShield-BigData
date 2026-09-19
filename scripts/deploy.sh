#!/usr/bin/env bash
# ==============================================================================
# CyberShield Production Automated Deployment Script
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo "🚀 CyberShield-BigData Automated Deployment & Verification"
echo "======================================================================"

# 1. التحقق من تثبيت Docker و Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ خطأ: Docker غير مثبت على هذا الخادم."
    exit 1
fi

# 2. بناء وتشغيل الحاويات في الخلفية
echo "🐳 بناء الحاويات وتشغيل الخدمات عبر Docker Compose..."
docker compose -f deployment/docker-compose.yml down --remove-orphans
docker compose -f deployment/docker-compose.yml up --build -d

# 3. فحص الجاهزية اللحظية (Health Check)
echo "⏳ انتظار جاهزية خدمة الـ REST API (Liveness Probe)..."
sleep 10

if curl -s -f http://localhost:8000/health/live > /dev/null; then
    echo "======================================================================"
    echo "🟢 اكتمل النشر بنجاح والخدمات تعمل بكفاءة:"
    echo "   - 📖 API Documentation: http://localhost:8000/docs"
    echo "   - 📊 SOC Master Dashboard: http://localhost:8501"
    echo "======================================================================"
else
    echo "❌ فشل فحص الصحة الأولي! تفقد السجلات عبر: docker compose -f deployment/docker-compose.yml logs"
    exit 1
fi