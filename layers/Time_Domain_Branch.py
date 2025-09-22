from layers.moving_average import MovingAvgPool

class TimeDomainBranch(nn.Module):
    """Dedicated time domain processing branch"""
    def __init__(self, seq_len: int, pred_len: int, n_features: int, kernel_sizes: list, 
                 individual: bool = True, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.kernel_sizes = kernel_sizes
        self.individual = individual
        self.dropout = dropout
        
        # Moving averages for trend extraction
        self.moving_avgs = nn.ModuleList([
            MovingAvgPool(kernel_size) for kernel_size in kernel_sizes
        ])
        
        # Time domain processing networks
        if individual:
            self.trend_processors = nn.ModuleList([
                nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(seq_len, seq_len // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(seq_len // 2, pred_len)
                    ) for _ in range(n_features)
                ]) for _ in kernel_sizes
            ])
            
            self.seasonal_processors = nn.ModuleList([
                nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(seq_len, seq_len // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(seq_len // 2, pred_len)
                    ) for _ in range(n_features)
                ]) for _ in kernel_sizes
            ])
        else:
            self.trend_processors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(seq_len, seq_len // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(seq_len // 2, pred_len)
                ) for _ in kernel_sizes
            ])
            
            self.seasonal_processors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(seq_len, seq_len // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(seq_len // 2, pred_len)
                ) for _ in kernel_sizes
            ])
        
        # Scale combination weights for different kernel sizes
        self.scale_weights = nn.Parameter(torch.ones(len(kernel_sizes)))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, n_features]
        Returns:
            time_output: [batch_size, pred_len, n_features]
        """
        batch_size = x.shape[0]
        scale_outputs = []
        
        for i, moving_avg in enumerate(self.moving_avgs):
            # Trend-seasonal decomposition
            trend = moving_avg(x)  # [batch_size, seq_len, n_features]
            seasonal = x - trend
            
            if self.individual:
                # Process each feature individually
                trend_out = torch.zeros([batch_size, self.pred_len, self.n_features], 
                                      dtype=x.dtype, device=x.device)
                seasonal_out = torch.zeros([batch_size, self.pred_len, self.n_features], 
                                         dtype=x.dtype, device=x.device)
                
                for j in range(self.n_features):
                    trend_out[:, :, j] = self.trend_processors[i][j](trend[:, :, j])
                    seasonal_out[:, :, j] = self.seasonal_processors[i][j](seasonal[:, :, j])
            else:
                # Process all features together
                trend_t = trend.permute(0, 2, 1)  # [batch_size, n_features, seq_len]
                seasonal_t = seasonal.permute(0, 2, 1)
                
                trend_out = self.trend_processors[i](trend_t).permute(0, 2, 1)
                seasonal_out = self.seasonal_processors[i](seasonal_t).permute(0, 2, 1)
            
            # Combine trend and seasonal
            scale_output = trend_out + seasonal_out
            scale_outputs.append(scale_output)
        
        # Weighted combination of different scales
        weights = F.softmax(self.scale_weights, dim=0)
        time_output = sum(w * out for w, out in zip(weights, scale_outputs))
        
        return time_output
