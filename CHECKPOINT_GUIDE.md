# 🔖 Checkpoint System Guide - نظام نقاط الحفظ

## ما هو Checkpoint System؟

نظام **نقاط الحفظ (Checkpointing)** يحفظ حالة كل مرحلة من مراحل الـ Pipeline بعد اكتمالها بنجاح. عند إعادة تشغيل البرنامج، يتم تخطي المراحل المكتملة والبدء من المرحلة التي توقف عندها.

---

## 🎯 الفوائد

✅ **توفير الوقت**: لا حاجة لإعادة تشغيل المراحل التي نجحت  
✅ **استئناف بعد الأخطاء**: إذا حدث خطأ في المرحلة 7، لن تحتاج لإعادة المراحل 1-6  
✅ **تجربة سريعة**: يمكنك تعديل كود مرحلة واحدة واختبارها بسرعة  
✅ **حفظ تلقائي**: يتم الحفظ تلقائياً بعد كل مرحلة ناجحة

---

## 📂 موقع ملفات Checkpoint

جميع ملفات حالة الـ Pipeline محفوظة في:
```
data/checkpoints/pipeline_state.json
```

---

## 🚀 أمثلة الاستخدام

### 1. تشغيل عادي (مع Checkpointing)
```bash
python main.py --mode all --data_path s3
```

- إذا كانت جميع المراحل مكتملة → يتم تخطيها كلها ✅
- إذا فشلت مرحلة → يبدأ منها في المرة القادمة
- إذا كانت المراحل 1-5 مكتملة والمرحلة 6 فشلت → سيبدأ من المرحلة 6

### 2. تعطيل Checkpoint (إعادة تشغيل كامل)
```bash
python main.py --mode all --data_path s3 --no-checkpoint
```
سيتم تشغيل **جميع المراحل من البداية** بدون فحص الحالة السابقة.

### 3. إعادة تعيين من مرحلة معينة
```bash
# إعادة تعيين من مرحلة Feature Engineering
python main.py --reset-from feature_engineering

# إعادة تعيين من مرحلة Model Selection
python main.py --reset-from model_selection
```
سيتم **حذف حالة هذه المرحلة وما بعدها** فقط، والمراحل السابقة ستبقى محفوظة.

### 4. حذف جميع Checkpoints
```bash
python main.py --clear-checkpoints
```
سيتم **مسح جميع الحالات** والبدء من الصفر في التشغيل القادم.

---

## 📋 أسماء المراحل

| الرقم | اسم المرحلة | Stage Name |
|------|-------------|------------|
| 1 | Ingestion | `ingestion` |
| 2 | Quality Gate | `quality` |
| 3 | Cleaning | `cleaning` |
| 4 | EDA | `eda` |
| 5 | Feature Engineering | `feature_engineering` |
| 6 | Feature Store & Splitting | `feature_store` |
| 7 | Model Selection | `model_selection` |
| 8 | Validation | `validation` |
| 9 | XAI | `xai` |
| 10 | Monitoring | `monitoring` |

---

## 🔍 فحص حالة Pipeline

يمكنك فتح ملف الحالة لمعرفة أي المراحل مكتملة:
```bash
cat data/checkpoints/pipeline_state.json
```

مثال على المحتوى:
```json
{
  "ingestion": {
    "completed": true,
    "timestamp": "2026-08-16T03:45:12.345678",
    "metadata": {
      "row_count": 128937,
      "columns": 79
    }
  },
  "quality": {
    "completed": true,
    "timestamp": "2026-08-16T03:46:30.123456",
    "metadata": {}
  },
  "cleaning": {
    "completed": true,
    "timestamp": "2026-08-16T03:47:45.987654",
    "metadata": {
      "row_count": 126945
    }
  }
}
```

---

## 🛠️ سيناريوهات الاستخدام

### السيناريو 1: خطأ في المرحلة 6
```bash
# التشغيل الأول - فشل في المرحلة 6
python main.py --mode all --data_path s3
# Error in stage 6!

# بعد تصليح الخطأ - سيبدأ من المرحلة 6 مباشرة
python main.py --mode all --data_path s3
# ✅ Stages 1-5 skipped (already completed)
# ▶️ Starting from stage 6...
```

### السيناريو 2: تعديل كود Feature Engineering
```bash
# حذف checkpoints من المرحلة 5 وما بعدها
python main.py --reset-from feature_engineering

# إعادة التشغيل - سيبدأ من المرحلة 5
python main.py --mode all --data_path s3
# ✅ Stages 1-4 skipped
# ▶️ Starting from stage 5...
```

### السيناريو 3: بداية جديدة تماماً
```bash
# حذف كل شيء
python main.py --clear-checkpoints

# تشغيل من الصفر
python main.py --mode all --data_path s3
# ▶️ All stages will run from scratch
```

---

## 💡 نصائح مهمة

### متى تستخدم `--no-checkpoint`؟
- عند تغيير البيانات المصدرية (dataset مختلف)
- عند تعديل configs أساسية
- عند الرغبة في قياس الأداء الكامل

### متى تستخدم `--reset-from`؟
- عند تعديل كود مرحلة محددة
- عند الرغبة في إعادة تدريب النموذج فقط
- عند تصليح bug في مرحلة متأخرة

### متى تستخدم `--clear-checkpoints`؟
- عند الانتقال لـ dataset جديد تماماً
- عند حدوث مشاكل في ملف الحالة
- قبل التسليم النهائي للمشروع

---

## ⚠️ ملاحظات مهمة

1. **البيانات المحفوظة**: بعض المراحل تحفظ بيانات في:
   - `data/processed/cicids2018_sample.parquet` (Ingestion cache)
   - `data/feature_store/nids_features_latest/` (Engineered features)
   - `data/feature_store/train_split/` (Training data)
   - `data/feature_store/val_split/` (Validation data)
   - `data/feature_store/test_split/` (Test data)

2. **الحذف اليدوي**: إذا حذفت هذه المجلدات يدوياً، قد تحتاج لحذف الـ checkpoint المقابل:
   ```bash
   python main.py --reset-from feature_store
   ```

3. **تحديث الكود**: عند تحديث كود مرحلة، استخدم `--reset-from` لإعادة تنفيذها.

---

## 🎉 الخلاصة

نظام Checkpoint يجعل التطوير أسرع بكثير! الآن يمكنك:
- ✅ تصليح الأخطاء دون إعادة تشغيل كل شيء
- ✅ تجربة تعديلات على مراحل محددة
- ✅ استئناف العمل بسهولة بعد انقطاع

**التشغيل الافتراضي دائماً مع Checkpoint مفعّل** - لا حاجة لأي flags إضافية! 🚀
