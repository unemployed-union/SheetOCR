import torch
import torch.nn as nn
from .patch_embedding import PatchEmbedding

class SimpleViTForOCR(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 img_height=112, 
                 img_width=448, 
                 patch_size=8, 
                 embed_dim=512,
                 in_channels=1, 
                 num_heads=8, 
                 num_layers=6,      # [변경] 12층 -> 6층 (데이터 2만 장에는 6층이 학습이 더 잘 됩니다)
                 dropout=0.1):
        super().__init__()

        # 높이/너비 패치 개수
        self.h_patches = img_height // patch_size  # 14
        self.w_patches = img_width // patch_size   # 56

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels, 
            patch_size=patch_size, 
            emb_size=embed_dim
        )

        # 2. Positional Embedding
        self.num_patches = self.h_patches * self.w_patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout, 
            activation='gelu', 
            batch_first=True,
            norm_first=True 
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # [핵심 추가] 세로 정보를 뭉개지 않고 '학습'해서 압축하는 층
        # 입력: (세로 14개 * 512차원) -> 출력: 512차원
        self.vertical_proj = nn.Linear(self.h_patches * embed_dim, embed_dim)
        self.act = nn.GELU()

        # 4. Final Norm & Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        # x: (Batch, 1, 112, 448)
        
        # 1. Patch Embedding & Pos
        x = self.patch_embed(x) 
        x = x + self.pos_embed
        
        # 2. Transformer
        x = self.encoder(x) # (Batch, 784, 512)
        
        # ---------------------------------------------------------
        # [핵심 변경] Flatten + Linear Projection
        # ---------------------------------------------------------
        B, N, C = x.shape
        
        # 1) 2D로 복원: (Batch, 14, 56, 512)
        x = x.view(B, self.h_patches, self.w_patches, C)
        
        # 2) 축 변경: (Batch, 56, 14, 512) -> 가로(Time)를 앞으로
        x = x.permute(0, 2, 1, 3)
        
        # 3) 세로와 채널을 합침(Flatten): (Batch, 56, 14*512)
        # 이제 위아래 정보가 섞이지 않고 나란히 펴짐
        x = x.reshape(B, self.w_patches, self.h_patches * C)
        
        # 4) Linear로 중요한 정보만 뽑아서 압축
        x = self.vertical_proj(x) # (Batch, 56, 512)
        x = self.act(x)

        # 3. Prediction
        x = self.norm(x)
        output = self.head(x) # (Batch, 56, Vocab)
        
        return output