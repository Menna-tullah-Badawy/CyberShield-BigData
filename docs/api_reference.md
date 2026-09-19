# 📡 API Reference — CyberShield NIDS API v3.0.0

Base URL: `http://localhost:8000` · Interactive docs: `/docs`

## `GET /health`

Liveness + model status.

```json
{"status": "healthy", "model": "Mamba SSM", "threats_in_kb": 6,
 "threats_detected": 0, "input": [10, 66]}
```

## `GET /api/metrics`

Batch-audit detection metrics.

```json
{"model": "Mamba SSM",
 "batch_test": {"TP": 50, "FP": 0, "TN": 50, "FN": 0, "ROC_AUC": 0.91},
 "threat_types_detected": 5}
```

## `GET /api/threats`

Full MITRE knowledge base + detected threat types with indicators and mitigations.

```json
{"threats": [{"threat_id": "MITRE-T1498", "title": "...", "tactic": "...",
  "playbook": "...", "iptables": ["..."], "indicators": "..."}],
 "detected": {"MITRE-T1498": {"count": 3, "hybrid_score": 0.9, "...": "..."}}}
```

## `POST /api/detect`

Real-time detection with XAI + RAG + response playbook.

Request:

```json
{"features": [[0.1, 0.2, "..."]], "src_ip": "192.168.1.10",
 "dst_port": 80, "protocol": "TCP"}
```

- `features`: `(SEQ_LEN × N_FEATURES)` matrix, e.g. `10 × 66`.
- Threat verdict requires `probability >= 0.03`.

Response (threat):

```json
{"incident_id": "INC-2026-NIDS-00001",
 "detection": {"is_threat": true, "probability": 0.97},
 "threat": {"id": "MITRE-T1498", "title": "...", "tactic": "...",
            "indicators": "...", "playbook": "..."},
 "xai": {"top_features": [{"name": "SYN_Flag_Cnt", "importance": 0.31}]},
 "rag": {"hybrid_score": 0.92},
 "response": {"iptables": ["iptables -A INPUT -s 192.168.1.10 -j DROP"]}}
```

Response (benign):

```json
{"detection": {"is_threat": false, "probability": 0.01}, "verdict": "BENIGN"}
```
