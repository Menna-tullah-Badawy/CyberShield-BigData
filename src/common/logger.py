"""
Centralized Structured Logging Module.
نظام التسجيل والمراقبة المركزي لتسجيل أحداث النظام في التيرمنال وملفات خارجية.
"""

import sys  # استيراد مكتبة النظام لإخراج الرسائل إلى مخرج التيرمنال القياسي
import os  # استيراد مكتبة نظام التشغيل لتحديد مسار ملف السجلات
import logging  # استيراد مكتبة التسجيل القياسية في بايثون
from logging.handlers import RotatingFileHandler  # استيراد معالج تدوير الملفات لمنع تضخم حجم السجل
import configs.settings as cfg  # استيراد ملف الإعدادات العامة للوصول لمسار مجلد الـ logs


def get_logger(name: str = "CyberShield") -> logging.Logger:
    """
    دالة لإنشاء وضبط كائن التسجيل (Logger) الموحد للنظام.
    :param name: اسم الموديول المستدعي للدالة لتسهيل تعقب مصدر الحدث.
    :return: كائن Logger مهيأ بالكامل.
    """
    # إنشاء أو جلب كائن الـ Logger بالاسم المحدد
    logger = logging.getLogger(name)
    
    # ضبط أدنى مستوى لتسجيل الأحداث (INFO لتسجيل العمليات الأساسية والأخطاء)
    logger.setLevel(logging.INFO)

    # منع تكرار إضافة المعالجات في حال تم استدعاء الـ Logger عدة مرات
    if not logger.handlers:
        
        # 1. تنسيق نص السجل: [التاريخ والوقت] [المستوى] [اسم الموديول]: نص الرسالة
        log_format = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 2. معالج إخراج السجلات في التيرمنال (Console Handler)
        console_handler = logging.StreamHandler(sys.stdout)
        # تعيين التنسيق لمعالج التيرمنال
        console_handler.setFormatter(log_format)
        # إضافة معالج التيرمنال للـ Logger الرئيسي
        logger.addHandler(console_handler)

        # 3. معالج كتابة السجلات في ملف خارجي مع التدوير التلقائي (File Handler)
        log_file_path = os.path.join(cfg.LOGS_DIR, "platform.log")
        # التدوير عند بلوغ الحجم 10 ميجابايت والاحتفاظ بـ 5 ملفات أرشيف كحد أقصى
        file_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8"
        )
        # تعيين التنسيق لمعالج الملفات
        file_handler.setFormatter(log_format)
        # إضافة معالج الملف للـ Logger الرئيسي
        logger.addHandler(file_handler)

    # إرجاع كائن الـ Logger الجاهز للاستخدام
    return logger