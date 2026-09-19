"""
MITRE ATT&CK Threat Mapping + Hybrid RAG Retrieval (BM25 + LSA).
نقل حرفي من Cell 4 في cybershield.ipynb.
"""

from typing import Dict, List

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

BM25_K1 = 1.5
BM25_B = 0.75
LSA_WEIGHT = 0.55
BM25_WEIGHT = 0.45


class BM25:
    """نفس كلاس BM25 من النوت بوك بالنص."""

    def __init__(self, corpus, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.n = len(corpus)
        self.dl = [len(d) for d in corpus]
        self.avg = np.mean(self.dl) if self.dl else 1
        self.tf = []
        self.df = {}
        for doc in corpus:
            tf = {}
            for w in doc:
                tf[w] = tf.get(w, 0) + 1
            self.tf.append(tf)
            for w in set(doc):
                self.df[w] = self.df.get(w, 0) + 1
        self.idf = {w: np.log((self.n - d + 0.5) / (d + 0.5) + 1)
                    for w, d in self.df.items()}

    def get_scores(self, q):
        s = np.zeros(self.n)
        for i in range(self.n):
            tf = self.tf[i]
            dl = self.dl[i]
            for w in q:
                if w in tf and w in self.idf:
                    f = tf[w]
                    idf = self.idf[w]
                    s[i] += idf * (f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg)))
        return s


# قاعدة التهديدات الستة بالنص من النوت بوك (MITRE + playbook + iptables)
THREAT_KB: List[Dict] = [
    {"threat_id": "MITRE-T1498", "title": "Network DoS SYN Flood DDoS", "tactic": "TA0040 - Impact",
     "keywords": "init win bytes forward packet length max flow duration syn flood ddos denial service",
     "playbook": "Deploy SYN cookies, drop asymmetric bursts, rate-limit TCP SYN via iptables.",
     "iptables": ["iptables -A INPUT -p tcp --syn -m limit --limit 1/s --limit-burst 3 -j ACCEPT",
                 "iptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP",
                 "iptables -A INPUT -m state --state INVALID -j DROP", "iptables -A INPUT -p tcp --syn -j DROP"],
     "indicators": "High SYN flag count, abnormal Init_Win_bytes, low Flow_IAT_Min, large Fwd_Packet_Length"},
    {"threat_id": "MITRE-T1046", "title": "Port Scan Discovery", "tactic": "TA0007 - Discovery",
     "keywords": "destination port flow iat minimum packet length variance scan sweep discovery nmap",
     "playbook": "Enforce strict perimeter firewall rules, block source IP.",
     "iptables": ["iptables -A INPUT -s {src_ip} -j DROP",
                 "iptables -A INPUT -p tcp --tcp-flags SYN,FIN SYN,FIN -j DROP",
                 "iptables -A INPUT -p tcp --dport 1:1024 -m connlimit --connlimit-above 5 -j DROP"],
     "indicators": "Multiple destination ports, low Flow_IAT, high packet length variance, scan patterns"},
    {"threat_id": "MITRE-T1041", "title": "Exfiltration Over C2", "tactic": "TA0010 - Exfiltration",
     "keywords": "forward bytes bulk backward bytes bulk flow bytes packets high volume outbound exfiltration",
     "playbook": "Block outbound connections, quarantine source IP.",
     "iptables": ["iptables -A OUTPUT -d {src_ip} -j DROP",
                 "iptables -A INPUT -s {src_ip} -m conntrack --ctstate ESTABLISHED,RELATED -j DROP",
                 "iptables -A FORWARD -s {src_ip} -j DROP"],
     "indicators": "High Fwd_Bytes_Bulk, abnormal Bwd_Bytes_Bulk, high flow bytes per second"},
    {"threat_id": "MITRE-T1110", "title": "SSH Brute Force", "tactic": "TA0006 - Credential Access",
     "keywords": "syn flag count ack flag count forward packets ssh port 22 brute force credential",
     "playbook": "Rate-limit SSH, block after failed attempts.",
     "iptables": ["iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --set --name SSH",
                 "iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --update --seconds 60 --hitcount 4 --name SSH -j DROP",
                 "iptables -A INPUT -p tcp --dport 22 -j ACCEPT"],
     "indicators": "High SYN+ACK flag count, repeated SSH port 22 connections, brute force patterns"},
    {"threat_id": "MITRE-T1059", "title": "Command Injection", "tactic": "TA0002 - Execution",
     "keywords": "flow packets per second high rate burst payload command injection reverse shell",
     "playbook": "Block reverse shell ports.",
     "iptables": ["iptables -A INPUT -s {src_ip} -p tcp --dport 4444 -j DROP",
                 "iptables -A INPUT -s {src_ip} -p tcp --dport 8080 -j DROP",
                 "iptables -A OUTPUT -d {src_ip} -j DROP"],
     "indicators": "High flow packets per second, burst payload patterns, reverse shell signatures"},
    {"threat_id": "MITRE-T1071", "title": "DNS Tunneling", "tactic": "TA0011 - C2",
     "keywords": "dns query length high flow duration long idle mean protocol udp port 53 tunneling",
     "playbook": "Block suspicious DNS traffic.",
     "iptables": ["iptables -A INPUT -s {src_ip} -p udp --dport 53 -j DROP",
                 "iptables -A OUTPUT -d {src_ip} -p udp --dport 53 -j DROP"],
     "indicators": "Long DNS queries, high flow duration, abnormal idle mean, UDP port 53"},
]


class HybridThreatRetriever:
    """BM25 + LSA (TF-IDF + SVD) بنفس أوزان النوت بوك 0.55/0.45."""

    def __init__(self, kb: List[Dict] = None):
        self.kb = kb or THREAT_KB
        kb_texts = [f"{d['title']} {d['keywords']} {d['playbook']}" for d in self.kb]
        self.bm25 = BM25([t.lower().split() for t in kb_texts])
        self.lsa_vec = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', ngram_range=(1, 2),
                                       sublinear_tf=True, max_features=500)
        lsa_mat = self.lsa_vec.fit_transform(kb_texts)
        n_comp = max(min(5, lsa_mat.shape[0] - 1, lsa_mat.shape[1] - 1), 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        lsa_emb = self.svd.fit_transform(lsa_mat)
        nrm = np.linalg.norm(lsa_emb, axis=1, keepdims=True)
        nrm[nrm == 0] = 1
        self.lsa_emb = lsa_emb / nrm

    def retrieve(self, query: str) -> Dict:
        """نفس دالة hybrid_retrieve من النوت بوك بالنص."""
        def norm(a):
            mn, mx = a.min(), a.max()
            return np.ones_like(a) * 0.5 if mx - mn < 1e-10 else (a - mn) / (mx - mn)

        b = norm(self.bm25.get_scores(query.lower().split()))
        q = self.lsa_vec.transform([query])
        ql = self.svd.transform(q)
        nn = np.linalg.norm(ql, axis=1, keepdims=True)
        nn[nn == 0] = 1
        ql = ql / nn
        l = norm((self.lsa_emb @ ql.T).flatten())
        h = LSA_WEIGHT * l + BM25_WEIGHT * b
        i = int(np.argmax(h))
        return {"doc": self.kb[i], "bm25_score": float(b[i]),
                "lsa_score": float(l[i]), "hybrid_score": float(h[i])}


class MitreAttackMapper(HybridThreatRetriever):
    """الاسم المعماري الأصلي للمسترجع الهجين (نفس السلوك)."""

    def map_threat(self, query: str) -> Dict[str, str]:
        doc = self.retrieve(query)["doc"]
        return {"threat_id": doc["threat_id"], "title": doc["title"], "tactic": doc["tactic"]}
