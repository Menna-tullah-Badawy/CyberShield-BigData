"""
Explainability Engine — full incident explanation: XAI saliency + RAG threat match.
يربط feature_importance (التدرجات) مع mitre_mapping (الاسترجاع الهجين) كما في Cell 4.
"""

from typing import Dict, List, Sequence

import torch

from src.explainability.feature_importance import GlobalFeatureImportance


class ExplainabilityEngine:
    """المنسق الشامل للتفسير الجنائي (نفس تدفق النوت بوك)."""

    def __init__(self, feature_names: Sequence[str] = None, retriever=None):
        self.saliency = GlobalFeatureImportance(feature_names)
        self._retriever = retriever

    @property
    def retriever(self):
        if self._retriever is None:
            from src.genai_reporting.mitre_mapping import HybridThreatRetriever

            self._retriever = HybridThreatRetriever()
        return self._retriever

    def generate_incident_explanation(
        self, model, x: torch.Tensor, feature_names: Sequence[str] = None, n: int = 5,
    ) -> Dict[str, object]:
        """تفسير الحادثة: أهم الميزات + أقرب تهديد MITRE (نفس سطور النوت بوك)."""
        names = list(feature_names) if feature_names else self.saliency.feature_names
        xf: List[Dict[str, object]] = self.saliency.extract_importance(model, x, names, n=n)
        query = " ".join(str(f["name"]) for f in xf).lower().replace("_", " ")
        rag = self.retriever.retrieve(query)
        return {"top_features": xf, "rag": rag,
                "threat": rag["doc"], "hybrid_score": rag["hybrid_score"]}
