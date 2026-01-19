import torch
import torch.nn as nn

import torch
import torch.nn as nn
from .patch_embedding import PatchEmbedding

class SimpleViTForOCR(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 img_height=112, 
                 img_width=448, 
                 patch_size=8,      # [핵심] 8x8 패치 (시력 강화)
                 embed_dim=512,     # [핵심] 512 차원 (뇌 용량 최적화)
                 in_channels=1,     # [핵심] 흑백 모드
                 num_heads=8,       # 512 / 64 = 8 (헤드 개수)
                 num_layers=12,     # [핵심] 12층 (깊은 사고)
                 dropout=0.1):
        super().__init__()

        # 1. Patch Embedding 연결
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels, 
            patch_size=patch_size, 
            emb_size=embed_dim
        )

        # 2. Positional Embedding (위치 정보)
        # 패치 개수 계산: (112 / 8) * (448 / 8) = 14 * 56 = 784개
        self.num_patches = (img_height // patch_size) * (img_width // patch_size)
        
        # 학습 가능한 위치 정보 파라미터 생성
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, # 2048
            dropout=dropout, 
            activation='gelu', 
            batch_first=True,
            norm_first=True  # [중요] Pre-Norm (학습이 훨씬 안정적임)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Final Norm & Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

        # 5. 가중치 초기화
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
        # 1. 이미지 -> 패치 벡터
        x = self.patch_embed(x) 
        
        # 2. 위치 정보 더하기 (순서 기억)
        # x shape: (Batch, 784, 512)
        x = x + self.pos_embed
        
        # 3. Transformer 통과 (깊은 추론)
        x = self.encoder(x)
        
        # 4. 최종 예측
        x = self.norm(x)
        output = self.head(x)
        
        return output