"""
Custom exceptions for data quality module.
الاستثناءات البرمجية المخصصة لطبقة فحص وتدقيق الجودة.
"""

from src.common.exceptions import DataQualityError  # وراثة كلاس أخطاء الجودة الأساسي في المنصة


class QualityError(DataQualityError):  # الكلاس الأساسي لأخطاء الجودة[cite: 5]
    """
    Base quality exception.
    استثناء عام لأي خلل يحدث داخل موديولات الجودة.
    """
    pass  # تعريف الكلاس كاستثناء مستقل[cite: 5]


class InvalidColumnError(QualityError):  # استثناء خاص بالأعمدة غير الموجودة[cite: 5]
    """
    Raised when a target column does not exist in the DataFrame.
    يتم إطلاقه عند محاولة فحص عمود غير مسجل في الـ Schema.
    """
    pass  # تعريف كلاس الاستثناء[cite: 5]


class InvalidBusinessKeyError(QualityError):  # استثناء خاص بمفاتيح الأعمال غير الصالحة[cite: 5]
    """
    Raised when business keys are invalid or empty.
    يتم إطلاقه إذا كانت قائمة المفاتيح المنطقية فارغة أو غير صحيحة.
    """
    pass  # تعريف كلاس الاستثناء[cite: 5]