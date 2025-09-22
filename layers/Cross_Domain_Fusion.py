import torch
import torch.nn as nn

class CrossDomainFusion(nn.Module):
    """Intelligent fusion of parallel time and frequency domain outputs"""
    def __init__(self, pred_len: int, n_features: int, dropout: float = 0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_features = n_features
        
        # Cross-attention between domains
        self.time_freq_attention = nn.MultiheadAttention(
            embed_dim=n_features, num_heads=min(4, n_features), batch_first=True, dropout=dropout
        )
        
        self.freq_time_attention = nn.MultiheadAttention(
            embed_dim=n_features, num_heads=min(4, n_features), batch_first=True, dropout=dropout
        )
        
        # Fusion networks
        self.domain_fusion = nn.Sequential(
            nn.Linear(n_features * 3, n_features * 4),  # time + freq + cross-attention
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_features * 4, n_features * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_features * 2, n_features)
        )
        
        # Adaptive domain weighting based on periodicity
        self.domain_gate = nn.Sequential(
            nn.Linear(n_features, n_features * 2),
            nn.ReLU(),
            nn.Linear(n_features * 2, n_features),
            nn.Sigmoid()
        )
        
        # Residual scaling
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, time_output: torch.Tensor, freq_output: torch.Tensor, 
                periodicity_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_output: [batch_size, pred_len, n_features]
            freq_output: [batch_size, pred_len, n_features] 
            periodicity_scores: [batch_size, n_features]
        Returns:
            fused_output: [batch_size, pred_len, n_features]
        """
        # Cross-domain attention
        time_attended, _ = self.time_freq_attention(time_output, freq_output, freq_output)
        freq_attended, _ = self.freq_time_attention(freq_output, time_output, time_output)
        
        # Combine all information
        combined = torch.cat([time_output, freq_output, time_attended + freq_attended], dim=-1)
        fused = self.domain_fusion(combined)
        
        # Adaptive domain weighting based on periodicity
        periodicity_gate = self.domain_gate(periodicity_scores.unsqueeze(1))  # [batch_size, 1, n_features]
        
        # Final output: adaptive combination with residual connection
        base_combination = (1 - periodicity_gate) * time_output + periodicity_gate * freq_output
        final_output = base_combination + self.residual_scale * fused
        
        return final_output
