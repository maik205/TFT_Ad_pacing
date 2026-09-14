"""
Temporal Fusion Transformer (TFT) Architecture for Multi-Horizon Ad Spend Forecasting.
Implements:
- Gated Linear Units (GLU)
- Gated Residual Networks (GRN)
- Variable Selection Networks (VSN)
- Interpretable Multi-Head Self-Attention
- Multi-Quantile Loss (Pinball Loss for q in {0.10, 0.50, 0.90})
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class GLU(nn.Module):
    """Gated Linear Unit (GLU) with sigmoid gating."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val, gate = self.fc(x).chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN):
    LayerNorm(a + GLU(W1*a + W2*c + b1))
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, context_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        residual = self.skip(x)
        h = self.fc1(x)
        if context is not None and self.context_dim is not None:
            h = h + self.context_fc(context)
        h = F.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = self.glu(h)
        return self.layer_norm(residual + h)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN):
    Learns dynamic instance-level weights over multiple input features.
    """
    def __init__(self, num_features: int, feature_dim: int, hidden_dim: int, context_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Per-feature transformation GRNs
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(feature_dim, hidden_dim, hidden_dim, context_dim=None, dropout=dropout)
            for _ in range(num_features)
        ])

        # Selection weight GRN
        self.weight_grn = GatedResidualNetwork(
            num_features * feature_dim, hidden_dim, num_features, context_dim=context_dim, dropout=dropout
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len, num_features, feature_dim) or (batch_size, num_features, feature_dim)
        orig_shape = x.shape
        flat_x = x.contiguous().view(*orig_shape[:-2], self.num_features * self.feature_dim)
        
        # Compute weights via Softmax
        weights = F.softmax(self.weight_grn(flat_x, context), dim=-1) # (..., num_features)
        
        # Transform individual features
        transformed = []
        for i in range(self.num_features):
            feat = x[..., i, :].contiguous()
            transformed.append(self.feature_grns[i](feat))
        
        transformed_stack = torch.stack(transformed, dim=-2).contiguous() # (..., num_features, hidden_dim)
        
        # Weighted combination: sum_j v_j * transformed_j
        weights_expanded = weights.unsqueeze(-1).contiguous()
        selected = (transformed_stack * weights_expanded).sum(dim=-2).contiguous()
        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention:
    Values are shared across attention heads so aggregated attention weights
    directly reflect interpretable historical importance.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        B, T_q, _ = q.shape
        _, T_k, _ = k.shape

        q_proj = self.q_proj(q.contiguous()).contiguous().view(B, T_q, self.n_heads, self.d_head).transpose(1, 2).contiguous()
        k_proj = self.k_proj(k.contiguous()).contiguous().view(B, T_k, self.n_heads, self.d_head).transpose(1, 2).contiguous()
        v_proj = self.v_proj(v.contiguous()).contiguous().view(B, T_k, self.n_heads, self.d_head).transpose(1, 2).contiguous()

        k_transposed = k_proj.transpose(-2, -1).contiguous()
        scores = torch.matmul(q_proj, k_transposed) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1).contiguous()
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v_proj).contiguous() # (B, n_heads, T_q, d_head)
        context = context.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        output = self.out_proj(context).contiguous()


        # Average attention weights across heads for interpretability
        avg_attn = attn_weights.mean(dim=1) # (B, T_q, T_k)
        return output, avg_attn


class TemporalFusionTransformer(nn.Module):
    """
    Multi-Horizon Temporal Fusion Transformer for Ad Spend Pacing.
    Predicts quantiles [0.10, 0.50, 0.90] across horizon tau_max.
    """
    def __init__(
        self,
        num_past_features: int = 5,
        num_future_features: int = 6,
        num_static_features: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        lookback: int = 168,
        horizon: int = 24,
        quantiles: List[float] = [0.10, 0.50, 0.90],
        dropout: float = 0.1
    ):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.quantiles = quantiles
        self.d_model = d_model

        # Linear projections for individual continuous features into feature_dim
        feat_embed_dim = 16
        self.past_projections = nn.ModuleList([nn.Linear(1, feat_embed_dim) for _ in range(num_past_features)])
        self.future_projections = nn.ModuleList([nn.Linear(1, feat_embed_dim) for _ in range(num_future_features)])
        self.static_projections = nn.ModuleList([nn.Linear(1, feat_embed_dim) for _ in range(num_static_features)])

        # Variable Selection Networks
        self.static_vsn = VariableSelectionNetwork(num_static_features, feat_embed_dim, d_model, dropout=dropout)
        self.past_vsn = VariableSelectionNetwork(num_past_features, feat_embed_dim, d_model, context_dim=d_model, dropout=dropout)
        self.future_vsn = VariableSelectionNetwork(num_future_features, feat_embed_dim, d_model, context_dim=d_model, dropout=dropout)

        # Static enrichment GRN
        self.static_enrichment = GatedResidualNetwork(d_model, d_model, d_model, context_dim=d_model, dropout=dropout)

        # Temporal Sequence Processing (LSTM)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.post_lstm_gate = GatedResidualNetwork(d_model, d_model, d_model, context_dim=d_model, dropout=dropout)

        # Interpretable Temporal Attention
        self.self_attn = InterpretableMultiHeadAttention(d_model, n_heads=n_heads, dropout=dropout)
        self.post_attn_gate = GatedResidualNetwork(d_model, d_model, d_model, context_dim=d_model, dropout=dropout)

        # Output multi-quantile forecast head
        self.output_layer = nn.Linear(d_model, len(quantiles))

    def forward(
        self,
        past_x: torch.Tensor,     # (B, lookback, num_past_features)
        future_x: torch.Tensor,   # (B, horizon, num_future_features)
        static_x: torch.Tensor    # (B, num_static_features)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = past_x.shape[0]

        # Project static features
        static_embedded = []
        for i, proj in enumerate(self.static_projections):
            static_embedded.append(proj(static_x[:, i:i+1]))
        static_stack = torch.stack(static_embedded, dim=1) # (B, num_static, feat_dim)
        static_context, static_weights = self.static_vsn(static_stack) # (B, d_model)

        # Project past features
        past_embedded = []
        for i, proj in enumerate(self.past_projections):
            past_embedded.append(proj(past_x[..., i:i+1]))
        past_stack = torch.stack(past_embedded, dim=-2).contiguous() # (B, lookback, num_past, feat_dim)
        
        # Use physical .repeat instead of virtual .expand so Metal allocates genuine memory buffers
        context_past = static_context.unsqueeze(1).repeat(1, self.lookback, 1).contiguous()
        past_selected, past_weights = self.past_vsn(past_stack, context=context_past)

        # Project future features
        future_embedded = []
        for i, proj in enumerate(self.future_projections):
            future_embedded.append(proj(future_x[..., i:i+1]))
        future_stack = torch.stack(future_embedded, dim=-2).contiguous() # (B, horizon, num_future, feat_dim)
        
        context_future = static_context.unsqueeze(1).repeat(1, self.horizon, 1).contiguous()
        future_selected, future_weights = self.future_vsn(future_stack, context=context_future)

        # Concatenate full timeline (lookback + horizon)
        temporal_sequence = torch.cat([past_selected, future_selected], dim=1).contiguous() # (B, lookback + horizon, d_model)

        # Static enrichment with physically allocated context
        c_e = static_context.unsqueeze(1).repeat(1, self.lookback + self.horizon, 1).contiguous()
        enriched = self.static_enrichment(temporal_sequence, context=c_e).contiguous()

        # Temporal LSTM pass
        lstm_out, _ = self.lstm(enriched)
        lstm_out = lstm_out.contiguous()
        lstm_gated = self.post_lstm_gate(lstm_out, context=c_e).contiguous()

        # Self-attention pass
        attn_out, attn_weights = self.self_attn(lstm_gated, lstm_gated, lstm_gated)
        attn_gated = self.post_attn_gate((attn_out + lstm_gated).contiguous(), context=c_e).contiguous()


        # Slice horizon forecast steps
        forecast_representation = attn_gated[:, self.lookback:, :].contiguous() # (B, horizon, d_model)
        
        # Quantile outputs: (B, horizon, len(quantiles))
        quantile_preds = self.output_layer(forecast_representation)

        interpretability_dict = {
            'static_weights': static_weights,
            'past_weights': past_weights,
            'future_weights': future_weights,
            'attn_weights': attn_weights
        }

        return quantile_preds, interpretability_dict


def quantile_pinball_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    quantiles: List[float] = [0.10, 0.50, 0.90]
) -> torch.Tensor:
    """
    Computes average Pinball Loss across specified quantiles.
    y_pred: (B, horizon, num_quantiles)
    y_true: (B, horizon)
    """
    total_loss = 0.0
    for i, q in enumerate(quantiles):
        err = y_true - y_pred[..., i]
        loss_q = torch.maximum(q * err, (q - 1.0) * err)
        total_loss = total_loss + loss_q.mean()
    return total_loss / len(quantiles)
