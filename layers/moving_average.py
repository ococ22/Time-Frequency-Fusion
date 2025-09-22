class MovingAvgPool(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super(MovingAvgPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        batch_size, seq_len, n_features = x.shape
        x = x.permute(0, 2, 1)
        total_padding = self.kernel_size - 1
        pad_left = total_padding // 2
        pad_right = total_padding - pad_left
        front = x[:, :, :1].repeat(1, 1, pad_left)
        end = x[:, :, -1:].repeat(1, 1, pad_right)
        x_padded = torch.cat([front, x, end], dim=2)
        x_pooled = F.avg_pool1d(x_padded, kernel_size=self.kernel_size, stride=self.stride)
        if x_pooled.shape[2] != seq_len:
            if x_pooled.shape[2] > seq_len:
                x_pooled = x_pooled[:, :, :seq_len]
            else:
                needed = seq_len - x_pooled.shape[2]
                pad_end = x_pooled[:, :, -1:].repeat(1, 1, needed)
                x_pooled = torch.cat([x_pooled, pad_end], dim=2)
        x_pooled = x_pooled.permute(0, 2, 1)
        return x_pooled
