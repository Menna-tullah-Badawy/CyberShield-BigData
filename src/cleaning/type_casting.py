"""
CyberShield-BigData: Column Data Type Casting Module
=====================================================

Safe and consistent Spark DataFrame type casting.
"""

from typing import Dict

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, expr

from src.common.logger import get_logger


logger = get_logger("TypeCasting")


class TypeCasting:
    """
    Safely cast Spark DataFrame columns.

    Numeric columns use try_cast so malformed values become NULL
    instead of crashing the Spark pipeline.

    Categorical columns such as label remain STRING.
    """

    CATEGORICAL_COLUMNS = {
        "label",
        "class",
        "category",
        "attack",
        "attack_type",
        "protocol_name",
        "flags",
    }

    def cast(
        self,
        df: DataFrame,
        column: str,
        dtype: str
    ) -> DataFrame:

        if column not in df.columns:
            logger.warning(
                f"⚠️ Column '{column}' does not exist. "
                f"Skipping type conversion."
            )
            return df

        target_type = dtype.strip().lower()

        # --------------------------------------------------------
        # Categorical columns must remain strings
        # --------------------------------------------------------

        if column.lower() in self.CATEGORICAL_COLUMNS:

            if target_type != "string":
                logger.warning(
                    f"⚠️ Categorical column '{column}' "
                    f"requested as '{target_type}'. "
                    f"Forcing STRING."
                )

            target_type = "string"

        logger.info(
            f"🧹 [Cleaning] Converting "
            f"[{column}] → [{target_type}]"
        )

        try:

            # String conversion
            if target_type == "string":
                return df.withColumn(
                    column,
                    col(column).cast("string")
                )

            # Safe conversion for numeric/other types
            return df.withColumn(
                column,
                expr(
                    f"try_cast(`{column}` AS {target_type})"
                )
            )

        except Exception as error:

            logger.error(
                f"❌ Failed to cast column "
                f"'{column}' to '{target_type}': "
                f"{error}"
            )

            return df

    def cast_multiple(
        self,
        df: DataFrame,
        type_mapping: Dict[str, str]
    ) -> DataFrame:

        logger.info(
            f"🧹 [Cleaning] Applying safe type conversions: "
            f"{type_mapping}"
        )

        transformed_df = df

        for column_name, target_type in type_mapping.items():

            transformed_df = self.cast(
                transformed_df,
                column_name,
                target_type
            )

        return transformed_df