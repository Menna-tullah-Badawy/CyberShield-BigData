"""
Distributed Categorical Encoding Module.
موديول ترميز المتغيرات الفئوية (StringIndexer & OneHotEncoder) المتوافق مع Spark ML.
"""

from typing import List, Tuple  # استيراد أدوات التوثيق النوعي
from pyspark.sql import DataFrame  # استيراد جدول بيانات سبارك
from pyspark.ml.feature import StringIndexer, OneHotEncoder  # استيراد أدوات الترميز في سبارك ML
from src.common.logger import get_logger  # استيراد المسجل

logger = get_logger(__name__)  # تهيئة المسجل


class CategoricalEncoder:  # تعريف كلاس الترميز الفئوي الموزع[cite: 18]
    """
    Encode categorical features into numeric indices and one-hot vectors.
    يحول النصوص والفئات إلى أرقام ومتجهات ثنائية لمنع انحياز النماذج.
    """

    def index_column(self, df: DataFrame, input_col: str, output_col: str = None) -> DataFrame:
        """
        تطبيق StringIndexer على عمود واحد لتحويله إلى فهرس رقمي.
        """
        if input_col not in df.columns:
            logger.warn(f"⚠️ Column '{input_col}' not found. Skipping indexing.")
            return df
        
        out_col = output_col if output_col else f"{input_col}_idx"
        logger.info(f"⚙️ [Feature Engineering] Indexing column: [{input_col}] -> [{out_col}]...")
        
        indexer = StringIndexer(inputCol=input_col, outputCol=out_col, handleInvalid="keep")
        return indexer.fit(df).transform(df)

    def apply_string_indexing(self, df: DataFrame, categorical_cols: List[str]) -> Tuple[DataFrame, List[str]]:
        """
        تطبيق StringIndexer لتحويل القيم الفئوية النصية إلى فهارس رقمية.
        """
        indexed_cols = []  # قائمة لتتبع أسماء الأعمدة بعد الفهرسة
        df_out = df  # جدول البيانات الناتج

        for col_name in categorical_cols:  # المرور على كل عمود فئوي
            if col_name in df_out.columns:  # فحص وجود العمود
                output_col = f"{col_name}_idx"  # اسم العمود الناتج
                logger.info(f"⚙️ [Feature Engineering] تطبيق StringIndexer على العمود: [{col_name}] -> [{output_col}]...")  # تسجيل الفهرسة
                
                # تهيئة المفهرس وتجاهل القيم الجديدة غير المرئية أثناء التدريب لتجنب الانهيار
                indexer = StringIndexer(inputCol=col_name, outputCol=output_col, handleInvalid="keep")  # إعداد المفهرس
                df_out = indexer.fit(df_out).transform(df_out)  # تدريب وتطبيق الفهرسة
                indexed_cols.append(output_col)  # إضافة اسم العمود المرمز للقائمة

        return df_out, indexed_cols  # إرجاع الجدول وقائمة الأعمدة المفهرسة

    def apply_one_hot_encoding(self, df: DataFrame, indexed_cols: List[str]) -> DataFrame:  # تطبيق الترميز الثنائي[cite: 18]
        """
        تطبيق One-Hot Encoding على الأعمدة المفهرسة لتحويلها إلى متجهات ثنائية مفرغة (Sparse Vectors).
        """
        if not indexed_cols:  # التحقق من توفر أعمدة مفهرسة
            return df

        output_cols = [f"{c}_ohe" for c in indexed_cols]  # تسمية الأعمدة بعد الترميز الثنائي
        logger.info(f"⚙️ [Feature Engineering] تطبيق OneHotEncoder على الأعمدة: {indexed_cols}...")  # تسجيل العملية

        # تهيئة المحول للترميز الثنائي المتعدد
        encoder = OneHotEncoder(inputCols=indexed_cols, outputCols=output_cols, dropLast=True)  # إعداد المحول وحذف العمود الأخير لتجنب التكرار[cite: 18]
        df_encoded = encoder.fit(df).transform(df)  # تنفيذ التحويل الموزع

        return df_encoded  # إرجاع الجدول بالمتجهات الثنائية