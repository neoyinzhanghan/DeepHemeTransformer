import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):
    def __init__(self, head_dim, use_flash_attention):
        super(Attn, self).__init__()
        self.head_dim = head_dim
        self.use_flash_attention = use_flash_attention

    def forward(self, q, k, v):
        if self.use_flash_attention:
            # Use PyTorch's built-in scaled dot product attention with flash attention support
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # Compute scaled dot product attention manually
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
                self.head_dim
            )
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_probs, v)
        return attn_output


class MultiHeadAttentionClassifier(nn.Module):
    def __init__(
        self,
        d_model=1024,
        num_heads=8,
        use_flash_attention=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_flash_attention = use_flash_attention

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection after attention
        self.out_proj = nn.Linear(d_model, d_model)

        # The attention mechanism
        self.attn = Attn(
            head_dim=self.head_dim, use_flash_attention=use_flash_attention
        )

    def forward(self, x):
        # Shape of x: (batch_size, N, d_model), where N is the sequence length

        batch_size = x.size(0)

        # Linear projections for Q, K, V (batch_size, num_heads, N+1, head_dim)
        q = (
            self.q_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply attention (batch_size, num_heads, N+1, head_dim)
        attn_output = self.attn(q, k, v)

        # Concatenate attention output across heads (batch_size, N+1, d_model)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        # Apply final linear projection
        x = self.out_proj(attn_output)

        return x


class DeepHemeTransformer(nn.Module):
    def __init__(self):
        super(DeepHemeTransformer, self).__init__()

        self.feature_projector = nn.Linear(2048, 1024)
        self.transformer = MultiHeadAttentionClassifier(d_model=1024, num_heads=8)
        self.last_layer_linear = nn.Linear(1024, 23)

    def forward(self, x):

        # x should be shaped like [b, N, 2048]
        batch_size = x.size(0)
        num_cells = x.size(1)

        # reshape x to [b * N, 2048]
        x = x.view(-1, 2048)
        assert (
            x.size(0) == batch_size * num_cells
        ), f"Checkpoint 0: x.size(0)={x.size(0)}, expected {batch_size * num_cells}"

        # project features to 1024
        x = self.feature_projector(x)
        assert x.size(1) == 1024, f"Checkpoint 1: x.size(1)={x.size(1)}, expected 1024"
        assert (
            x.size(0) == batch_size * num_cells
        ), f"Checkpoint 2: x.size(0)={x.size(0)}, expected {batch_size * num_cells}"

        # reshape x to [b, N, 1024]
        x = x.view(batch_size, num_cells, 1024)
        assert (
            x.size(0) == batch_size
        ), f"Checkpoint 3: x.size(0)={x.size(0)}, expected {batch_size}"
        assert (
            x.size(1) == num_cells
        ), f"Checkpoint 4: x.size(1)={x.size(1)}, expected {num_cells}"

        # pass through transformer
        x = self.transformer(x)
        assert (
            x.size(0) == batch_size
        ), f"Checkpoint 5: x.size(0)={x.size(0)}, expected {batch_size}"
        assert (
            x.size(1) == num_cells
        ), f"Checkpoint 6: x.size(1)={x.size(1)}, expected {num_cells}"
        assert x.size(2) == 1024, f"Checkpoint 7: x.size(2)={x.size(2)}, expected 1024"

        # reshape x to [b * N, 1024]
        x = x.view(-1, 1024)
        assert (
            x.size(0) == batch_size * num_cells
        ), f"Checkpoint 8: x.size(0)={x.size(0)}, expected {batch_size * num_cells}"

        # pass through final linear layer
        x = self.last_layer_linear(x)
        assert (
            x.size(0) == batch_size * num_cells
        ), f"Checkpoint 9: x.size(0)={x.size(0)}, expected {batch_size * num_cells}"

        # reshape x to [b, N, 23]
        x = x.view(batch_size, num_cells, 23)
        assert (
            x.size(0) == batch_size
        ), f"Checkpoint 10: x.size(0)={x.size(0)}, expected {batch_size}"

        return x
