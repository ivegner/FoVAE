import torch
import torch.nn as nn

# import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class VisionTransformer(nn.Module):
    # adapted from https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/11-vision-transformer.html
    # based on ViT: https://arxiv.org/abs/2010.11929
    def __init__(
        self,
        input_dim,
        output_dim,
        embed_dim,
        hidden_dim,
        # num_channels,
        num_heads,
        num_layers,
        # out_dim,
        # patch_dim,
        do_flag_last_step: bool = False,
        dropout=0.0,
    ):
        """
        Inputs:
            input_dim - Dimensionality of the input feature vectors
            output_dim - Dimensionality of the output feature vectors
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            do_flag_last_step - Whether to apply a special embedding to the last step
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        # Layers/Networks
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_layer = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.Sequential(
            *(
                AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            )
        )
        self.next_token_predictor = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, output_dim)
        )
        self.dropout = nn.Dropout(dropout)

        self.do_flag_last_step = do_flag_last_step
        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, 1 + max_num_patches, embed_dim))
        self.last_step_embedding = nn.Parameter(torch.randn(embed_dim))


    def forward(self, patches: torch.Tensor, mask: torch.Tensor = None):
        # Preprocess input
        B, T, dim = patches.shape
        assert dim == self.input_dim, f"Expected input dimension {self.input_dim}, got {dim}"
        if mask is not None:
            assert (
                mask.shape == (B, T)
            ), f"Expected mask shape {(B, T)}, got {mask.shape}"
            masked_patches = patches * mask[:, :, None]
        else:
            masked_patches = patches

        # reshapes workaround for https://github.com/pytorch/pytorch/issues/95883
        x = self.input_layer(masked_patches.reshape(B * T, dim)).reshape(B, T, -1)

        # Add CLS token
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        # x = x + self.pos_embedding[:, : T + 1]

        if self.do_flag_last_step:
            x[:, -1] = x[:, -1] + self.last_step_embedding

        # # Apply Transformer
        x = self.dropout(x)
        # x = x.transpose(0, 1)
        x = self.transformer(x)

        # # Perform next-token prediction
        cls = x[:, 0]
        out = self.next_token_predictor(cls)

        return out
