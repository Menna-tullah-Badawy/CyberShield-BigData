"""
Workflow Execution and Task State Manager.
مدير تتبع حالات تنفيذ المهام ومراقبة الأخطاء وإعادة المحاولة التلقائية.
"""

from typing import Dict, Any, Callable
import time
from src.common.logger import get_logger

logger = get_logger(__name__)


class WorkflowManager:
    """
    Manages task execution DAGs, stage timing, error recovery, and pipeline telemetry states.
    """

    def __init__(self, workflow_name: str = "CyberShield_Workflow"):
        self.workflow_name = workflow_name
        self.task_history = []

    def execute_stage(self, stage_name: str, stage_callable: Callable, *args, **kwargs) -> Any:
        """تنفيذ مرحلة برمجية مع قياس الزمن وتتبع حالة النجاح أو الفشل."""
        logger.info(f"▶️ [Workflow] بدء تنفيذ المرحلة: [{stage_name}]...")
        start_time = time.time()
        
        try:
            result = stage_callable(*args, **kwargs)
            duration = round(time.time() - start_time, 3)
            self.task_history.append({"stage": stage_name, "status": "SUCCESS", "duration_sec": duration})
            logger.info(f"✅ [Workflow] اكتملت المرحلة [{stage_name}] بنجاح في {duration}s.")
            return result
        except Exception as err:
            duration = round(time.time() - start_time, 3)
            self.task_history.append({"stage": stage_name, "status": "FAILED", "duration_sec": duration, "error": str(err)})
            logger.error(f"❌ [Workflow] فشلت المرحلة [{stage_name}] بعد {duration}s: {str(err)}")
            raise err

    def get_summary(self) -> Dict[str, Any]:
        """إرجاع تقرير ملخص سير العمليات."""
        total_time = sum(t["duration_sec"] for t in self.task_history)
        all_passed = all(t["status"] == "SUCCESS" for t in self.task_history)
        return {
            "workflow_name": self.workflow_name,
            "overall_status": "COMPLETED" if all_passed else "FAILED",
            "total_execution_time_sec": round(total_time, 3),
            "stages_executed": self.task_history
        }