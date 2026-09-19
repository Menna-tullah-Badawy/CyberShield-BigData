"""
Deep Learning Architectures Suite for Temporal NIDS Benchmarking.
يضم نماذج: Bi-LSTM, Bi-GRU, والمعمارية الحديثة Mamba (Selective State Space Model).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class BiGRUModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class SelectiveSSMBlock(nn.Module):
    """
    Selective State Space Model Block (Mamba Mechanism) with O(N) Linear Complexity.
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)))
        self.D = nn.Parameter(torch.ones(d_model))

        self.x_proj = nn.Linear(d_model, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_model, bias=True)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = u.shape
        A = -torch.exp(self.A_log)

        proj = self.x_proj(u)
        B = proj[:, :, :self.d_state]
        C = proj[:, :, self.d_state:2*self.d_state]
        dt = F.softplus(self.dt_proj(proj[:, :, -1:]))

        y = torch.zeros_like(u)
        h = torch.zeros(batch_size, d_model, self.d_state, device=u.device)

        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1)
            u_t = u[:, t, :].unsqueeze(-1)
            B_t = B[:, t, :].unsqueeze(1)
            C_t = C[:, t, :].unsqueeze(-1)

            dA_t = torch.exp(dt_t * A.unsqueeze(0))
            dB_t = dt_t * B_t

            h = h * dA_t + dB_t * u_t
            y[:, t, :] = (torch.matmul(h, C_t).squeeze(-1) + self.D * u[:, t, :])

        return y


class MambaNIDS(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, d_state: int = 16, num_layers: int = 2, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ssm": SelectiveSSMBlock(d_model=d_model, d_state=d_state),
                "norm": nn.LayerNorm(d_model),
                "drop": nn.Dropout(dropout)
            }) for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for layer in self.layers:
            residual = h
            out = layer["ssm"](layer["norm"](h))
            h = residual + layer["drop"](out)
        out_pooled = h.mean(dim=1)
        return self.classifier(out_pooled)


class CNNBiLSTMModel(nn.Module):
    """
    CNN + BiLSTM with Attention pooling.
    منقول من Cell 1 في cybershield.ipynb (كان معرّفاً داخل النوت بوك فقط).
    """
    def __init__(self, input_dim, cnn_out=64, hidden_dim=64, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, cnn_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_out)
        self.lstm = nn.LSTM(cnn_out, hidden_dim, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, 32), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(32, 2))

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x.permute(0, 2, 1)))).permute(0, 2, 1)
        out, _ = self.lstm(h)
        w = F.softmax(self.attn(out), dim=1)
        return self.fc(torch.sum(out * w, dim=1))


class SequenceClassifier(nn.Module):
    """واجهة موحدة لاختيار المعمارية."""
    def __init__(self, arch_type: str, input_dim: int, num_classes: int = 2, **kwargs):
        super().__init__()
        arch_type = arch_type.lower()
        if arch_type == "bilstm":
            self.model = BiLSTMModel(input_dim=input_dim, num_classes=num_classes, **kwargs)
        elif arch_type == "bigru":
            self.model = BiGRUModel(input_dim=input_dim, num_classes=num_classes, **kwargs)
        elif arch_type == "mamba":
            self.model = MambaNIDS(input_dim=input_dim, num_classes=num_classes, **kwargs)
        elif arch_type in ("cnn_bilstm", "cnn-bilstm", "cnnbilstm"):
            self.model = CNNBiLSTMModel(input_dim=input_dim, **kwargs)
        else:
            raise ValueError(f"المعمارية '{arch_type}' غير معرفة.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)