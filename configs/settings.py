import os  # استيراد مكتبة التعامل مع نظام التشغيل والمسارات
from pathlib import Path  # استيراد أداة المسارات الحديثة والمرنة في بايثون

# ==============================================================================
# 1. تحديد المسارات الأساسية للنظام (Directory Paths Configuration)
# ==============================================================================

# تحديد المسار الجذري للمشروع بالكامل بالرجوع مستويين للوراء من موقع هذا الملف
BASE_DIR = Path(__file__).resolve().parent.parent

# مسار مجلد البيانات الرئيسي
DATA_DIR = os.path.join(BASE_DIR, "data")
# مسار البيانات الخام غير المعدلة (Raw Layer)
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
# مسار البيانات النظيفة المخزنة بصيغة باركيه (Processed Layer)
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
# مسار الخصائص والميزات المستخرجة (Features Layer)
FEATURES_DATA_DIR = os.path.join(DATA_DIR, "features")
# مسار نقاط استرجاع تدفقات البث الحي (Streaming Checkpoints)
CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")

# مسار مجلد تصدير التقارير الإحصائية والجودة
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
# مسار مجلد تصدير سجلات التشغيل (Logs)
LOGS_DIR = os.path.join(BASE_DIR, "logs")
# مسار مجلد تصدير وحفظ نماذج الذكاء الاصطناعي المدربة
MODELS_DIR = os.path.join(BASE_DIR, "export", "models")

# التأكد من إنشاء المجلدات تلقائياً في حال عدم وجودها لمنع حدوث أخطاء تشغيل
for directory_path in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FEATURES_DATA_DIR,
    CHECKPOINTS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    MODELS_DIR,
]:
    # إنشاء المجلد مع تجاهل الخطأ إذا كان موجوداً مسبقاً وإنشاء المجلدات الأبوية إن لزمت
    os.makedirs(directory_path, exist_ok=True)


# ==============================================================================
# 2. إعدادات محرك الحوسبة الموزعة (Apache Spark Engine Configurations)
# ==============================================================================

# اسم التطبيق المعروض في لوحة تحكم سبارك (Spark UI)
SPARK_APP_NAME = "CyberShield-Enterprise-NIDS"
# نمط التشغيل الموزع (local[*] تعني استغلال جميع الأنوية المتاحة في المعالج)
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
# تخصيص حجم ذاكرة الوصول العشوائي للـ Driver
SPARK_DRIVER_MEMORY = "8g"
# تخصيص حجم ذاكرة الوصول العشوائي لكل منفذ (Executor)
SPARK_EXECUTOR_MEMORY = "8g"
# عدد أقسام التوزيع عند خلط وإعادة ترتيب البيانات (Shuffle Partitions)
SPARK_SHUFFLE_PARTITIONS = "200"


# ==============================================================================
# 3. إعدادات الشبكة والأمن السيبراني (Cybersecurity Traffic & Feature Settings)
# ==============================================================================

# قائمة الأعمدة الرقمية الرئيسية لحزم الشبكة
NUMERICAL_FEATURES = ["Packet_Length", "Time_Delta", "Header_Length", "Window_Size"]

# قائمة الأعمدة الفئوية المراد ترميزها
CATEGORICAL_FEATURES = ["Protocol", "Flags"]

# العمود المستهدف للتصنيف (0 لحركة المرور الطبيعية، 1 للهجمات الخبيثة)
TARGET_COLUMN = "label"

# نسبة تقسيم البيانات الصارمة (60% تدريب - 20% تحقق - 20% اختبار)
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# البذرة العشوائية لضمان إمكانية إعادة إنتاج نفس النتائج الرياضية (Reproducibility)
RANDOM_SEED = 42