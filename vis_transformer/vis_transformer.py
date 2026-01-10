import torch
import torch.nn as nn
from .patch_embedding import PatchEmbedding

class SimpleViTForOCR(nn.Module):
    def __init__(self, vocab_size, img_height=112, img_width=448, patch_size=None, embed_dim=768):
        super().__init__()

        # 1. CNN Embedding (Hybrid)
        self.patch_embed = PatchEmbedding()

        # 3. Transformer Encoder (Pre-Norm 적용!)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=8, 
            dim_feedforward=4096, 
            dropout=0.1, 
            activation='gelu', # ReLU보다 부드러운 GELU 추천
            batch_first=True,
            norm_first=True  # <--- [핵심 1] 학습 안정성의 열쇠!
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # 4. [핵심 2] 최종 LayerNorm
        self.norm = nn.LayerNorm(embed_dim)

        # 5. Head
        self.head = nn.Linear(embed_dim, vocab_size)

        # 6. [핵심 3] 가중치 초기화 (자동 실행)
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
        # 1. CNN & Reshape
        x = self.patch_embed(x) 
        
        # 3. Transformer
        x = self.encoder(x)
        
        # 4. Normalize & Predict
        x = self.norm(x) # Head에 들어가기 전에 단정하게 정리
        output = self.head(x)
        
        return output