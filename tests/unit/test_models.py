"""
Unit Tests for Candidate Classifier Models & Hyperparameter Builders.
"""

from src.models.candidate_models import CandidateModelFactory
from src.models.hyperparameter_tuner import SparkHyperparameterTuner


def test_candidate_model_factory():
    """اختبار إنشاء عائلات الخوارزميات المرشحة للمنافسة."""
    models = CandidateModelFactory.get_candidate_models("final_features", "label")
    
    assert "Logistic_Regression" in models
    assert "Random_Forest" in models
    assert "GBT_Classifier" in models


def test_hyperparameter_grid_builder():
    """اختبار بناء شبكات المعاملات الفائقة لكل خوارزمية."""
    models = CandidateModelFactory.get_candidate_models("final_features", "label")
    rf_model = models["Random_Forest"]
    
    grid = SparkHyperparameterTuner.build_param_grid("Random_Forest", rf_model)
    assert len(grid) > 0
    assert len(grid) == 4 # 2 numTrees * 2 maxDepth