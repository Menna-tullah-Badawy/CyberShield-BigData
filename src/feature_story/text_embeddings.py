"""
Distributed SecBERT Text Embeddings Module.
محرك استخراج التضمينات السياقية لحمولات حزم الشبكة عبر SecBERT الموزع بـ PyArrow Pandas UDFs.
"""

from typing import List  # استيراد أدوات التوثيق النوعي
import pandas as pd  # استيراد مكتبة بانداس للتعامل مع السلاسل النصية في UDF
import torch  # استيراد مكتبة PyTorch لتشغيل الشبكة العصبية
from transformers import AutoTokenizer, AutoModel  # استيراد أدوات HuggingFace
from pyspark.sql import DataFrame  # استيراد نوع جدول بيانات سبارك
from pyspark.sql.functions import pandas_udf, col  # استيراد دالة UDF الموزعة
from pyspark.sql.types import ArrayType, FloatType  # استيراد نوع المتجه الناتج (قائمة من أرقام عشرية)
from src.common.logger import get_logger  # استيراد نظام التسجيل
from src.common.decorators import time_execution  # استيراد مصمم قياس زمن التنفيذ

logger = get_logger(__name__)  # تهيئة المسجل لهذا الموديول


class SecBERTFeatureExtractor:
    """
    Distributed NLP Payload Embedding Extractor using SecBERT.
    يقوم بتحويل النصوص والأوامر البرمجية في الحمولات إلى متجهات أرقام دلالية (Semantic Embeddings).
    """

    def __init__(self, model_name: str = "jackaduma/SecBERT", max_length: int = 64, device: str = None):
        self.model_name = model_name  # تحديد اسم النموذج المتخصص في الأمن السيبراني
        self.max_length = max_length  # تحديد أقصى طول مسموح به للرموز (Tokens)
        # تحديد جهاز المعالجة (GPU إذا توفر، وإلا فالاعتماد على المعالج المركزي CPU)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🧠 [SecBERT] تهيئة نموذج التضمينات الأمنية [{self.model_name}] على جهاز: [{self.device}]")

    def _build_embedding_udf(self):
        """
        بناء دالة موزعة (Pandas UDF) لتوزيع عملية التضمين على كافة عقد المعالجة (Workers) في سبارك.
        """
        model_name = self.model_name  # تمرير المتغيرات لدالة الـ Worker
        max_length = self.max_length  # أقصى طول للرموز
        device = self.device  # جهاز المعالجة

        @pandas_udf(ArrayType(FloatType()))  # تحديد نوع الخرج كمصفوفة من الأرقام العشرية
        def extract_payload_embeddings_udf(payload_series: pd.Series) -> pd.Series:
            """
            دالة داخلية تنفذ على كل دفعة (Batch) داخل عمال الـ Spark Cluster.
            """
            # تحميل النموذج والمحلل اللغوي محلياً داخل كل منفذ (Executor)
            tokenizer = AutoTokenizer.from_pretrained(model_name)  # تحميل الـ Tokenizer
            model = AutoModel.from_pretrained(model_name)  # تحميل أوزان النموذج
            model.to(device)  # نقل النموذج للذاكرة المناسبة
            model.eval()  # وضع النموذج في وضع الاستدلال (Inference Mode)

            embeddings_list = []  # قائمة لتجميع المتجهات الناتجة

            # معالجة النصوص كدفعات لتسريع زمن المعالجة
            with torch.no_grad():  # تعطيل حساب التدرجات لتوفير الذاكرة وتسريع التنفيذ
                # استبدال القيم الفارغة بنصوص افتراضية
                sanitized_texts = payload_series.fillna("EMPTY_PAYLOAD").astype(str).tolist()
                
                # تقطيع وترميز النصوص وتحويلها إلى متجهات PyTorch
                encoded_inputs = tokenizer(
                    sanitized_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(device)

                # تمرير المدخلات داخل المحول واستخراج التمثيلات المخفية (Hidden States)
                outputs = model(**encoded_inputs)
                
                # استخراج تمثيل رمز البداية [CLS] الذي يمثل المعنى الدلالي الكامل للحمولة
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                # تحويل النتائج إلى قوائم أرقام عشرية
                for emb in cls_embeddings:
                    embeddings_list.append(emb.tolist())

            return pd.Series(embeddings_list)  # إرجاع سلسلة المتجهات الموزعة

        return extract_payload_embeddings_udf

    @time_execution  # قياس زمن تنفيذ استخراج التضمينات
    def transform_payloads(self, df: DataFrame, payload_col: str = "payload", output_col: str = "secbert_embedding") -> DataFrame:
        """
        تطبيق استخراج التضمينات الدلالية على عمود الحمولة وإضافته لجدول بيانات Spark.
        """
        if payload_col not in df.columns:  # التحقق من وجود عمود الحمولة
            logger.warning(f"⚠️ عمود الحمولة '{payload_col}' غير موجود، تم تخطي تضمينات SecBERT.")  # تسجيل تحذير
            return df  # إعادة الجدول الأصلي

        logger.info(f"🚀 [SecBERT] بدء التوليد الموزع للتضمينات السياقية لعمود [{payload_col}]...")  # تسجيل بدء التوليد
        
        # بناء الـ UDF الموزعة
        embedding_udf = self._build_embedding_udf()
        
        # تطبيق الدالة الموزعة على العمود وإرجاع الجدول المحدث
        transformed_df = df.withColumn(output_col, embedding_udf(col(payload_col)))
        
        logger.info(f"✅ تم توليد تضمينات SecBERT بنجاح وإضافتها للعمود: [{output_col}].")  # تسجيل اكتمال العملية
        return transformed_df