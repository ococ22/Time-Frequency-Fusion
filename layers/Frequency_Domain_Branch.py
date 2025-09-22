class FrequencyDomainBranch(nn.Module):
    """Dedicated frequency domain processing branch"""
    def __init__(self, seq_len: int, pred_len: int, n_features: int, 
                 n_freq_bands: int = 8, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.n_freq_bands = n_freq_bands
        
        # Frequency band analyzers
        self.freq_band_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, 32),  # Real and imaginary parts
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)   # Output real and imaginary
            ) for _ in range(n_freq_bands)
        ])
        
        # Frequency domain predictor
        self.freq_predictor = nn.Sequential(
            nn.Linear(seq_len, seq_len // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len // 2, pred_len)
        )
        
        # Periodicity and seasonality detector
        self.periodicity_net = nn.Sequential(
            nn.Linear(n_freq_bands, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_features),
            nn.Sigmoid()
        )
        
        # Create frequency band ranges
        self.register_buffer('freq_bands', self._create_freq_bands())
        
    def _create_freq_bands(self) -> torch.Tensor:
        """Create frequency band ranges"""
        nyquist = self.seq_len // 2 + 1
        band_size = max(1, nyquist // self.n_freq_bands)
        bands = []
        
        for i in range(self.n_freq_bands):
            start = i * band_size
            end = min((i + 1) * band_size, nyquist)
            if start < end:
                bands.append([start, end])
        
        return torch.tensor(bands)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, n_features]
        Returns:
            freq_output: [batch_size, pred_len, n_features]
            periodicity_scores: [batch_size, n_features]
        """
        batch_size, seq_len, n_features = x.shape
        freq_outputs = []
        all_band_features = []
        
        for feat_idx in range(n_features):
            feat_data = x[:, :, feat_idx]  # [batch_size, seq_len]
            
            # FFT
            x_fft = torch.fft.rfft(feat_data, dim=1)  # [batch_size, seq_len//2+1]
            enhanced_fft = x_fft.clone()
            band_features = []
            
            # Process each frequency band
            for band_idx, (start, end) in enumerate(self.freq_bands):
                if start < x_fft.shape[1] and end <= x_fft.shape[1] and start < end:
                    # Extract frequency band
                    band_fft = x_fft[:, start:end]
                    
                    if band_fft.shape[1] > 0:
                        # Convert to real/imag representation
                        band_real_imag = torch.stack([band_fft.real, band_fft.imag], dim=-1)
                        
                        # Process through band processor
                        band_shape = band_real_imag.shape
                        band_flat = band_real_imag.view(-1, 2)
                        enhanced_flat = self.freq_band_processors[band_idx](band_flat)
                        enhanced_band = enhanced_flat.view(band_shape)
                        
                        # Convert back to complex
                        enhanced_band_complex = torch.complex(enhanced_band[..., 0], enhanced_band[..., 1])
                        enhanced_fft[:, start:end] = enhanced_band_complex
                        
                        # Store band features for periodicity analysis
                        band_magnitude = torch.abs(band_fft).mean(dim=1)  # [batch_size]
                        band_features.append(band_magnitude)
            
            # Pad band_features if necessary
            while len(band_features) < self.n_freq_bands:
                band_features.append(torch.zeros(batch_size, device=x.device))
            
            all_band_features.append(torch.stack(band_features[:self.n_freq_bands], dim=1))
            
            # Convert back to time domain and predict
            enhanced_time = torch.fft.irfft(enhanced_fft, n=seq_len, dim=1)
            freq_pred = self.freq_predictor(enhanced_time)  # [batch_size, pred_len]
            freq_outputs.append(freq_pred)
        
        # Stack outputs
        freq_output = torch.stack(freq_outputs, dim=2)  # [batch_size, pred_len, n_features]
        
        # Compute periodicity scores
        if all_band_features:
            band_features_tensor = torch.stack(all_band_features, dim=2)  # [batch_size, n_bands, n_features]
            # Average across features for periodicity detection
            avg_band_features = band_features_tensor.mean(dim=2)  # [batch_size, n_bands]
            periodicity_scores = self.periodicity_net(avg_band_features)  # [batch_size, n_features]
        else:
            periodicity_scores = torch.zeros(batch_size, n_features, device=x.device)
        
        return freq_output, periodicity_scores
