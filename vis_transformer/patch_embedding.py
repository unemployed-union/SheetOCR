import torch
import torch.nn as nn
import math

import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=8, emb_size=512):
        super().__init__()
        self.patch_size = patch_size
        
        # [핵심] 
        # 1. in_channels=1 (흑백)
        # 2. patch_size=8 (고해상도, 작은 글자 인식용)
        # 3. emb_size=512 (적절한 벡터 크기)
        self.proj = nn.Conv2d(
            in_channels, 
            emb_size, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        # x: (Batch, 1, 112, 448)
        x = self.proj(x)  # -> (Batch, 512, 14, 56) (112/8=14, 448/8=56)
        
        # (Batch, 512, 14, 56) -> (Batch, 512, 784) -> (Batch, 784, 512)
        x = x.flatten(2).transpose(1, 2)
        return x

class SimpleViTForOCR(nn.Module):
    def __init__(self, vocab_size, embed_dim=384, num_heads=8, num_layers=6):
        super().__init__()
        
        # 1. 임베딩 층 (위에서 정의한 클래스)
        self.patch_embed = PatchEmbedding(emb_size=embed_dim)
        
        # 2. 트랜스포머 인코더 (PyTorch 내장 모듈 조립)
        # 직접 구현하고 싶다면 nn.TransformerEncoderLayer 내부를 뜯어야 하지만,
        # 학습용으로는 이 블록을 '사용'하는 것부터 시작하는 게 좋습니다.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim*4, # 보통 4배 확장
            dropout=0.1,
            activation='gelu', # ViT는 ReLU보다 GELU를 선호
            batch_first=True   # (Batch, Seq, Dim) 순서 유지
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 정규화 (LayerNorm) - 인코더 끝나고 한 번 해주는 게 국룰
        self.norm = nn.LayerNorm(embed_dim)
        
        # 4. 출력층 (헤드)
        self.head = nn.Linear(embed_dim, vocab_size)

        # (중요) 가중치 초기화: 처음부터 학습하는 거라 초기값이 중요함
        self._init_weights()

    def _init_weights(self):
        # Xavier/Kaiming 초기화 (학습 잘 되게 도와줌)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # 1. 임베딩 (이미지 -> 벡터 시퀀스)
        x = self.patch_embed(x)
        
        # 2. 트랜스포머 통과
        x = self.transformer_encoder(x)
        x = self.norm(x)
        
        # 3. [CLS] 토큰 제거 (OCR은 순서가 중요하므로 요약본인 맨 앞 토큰은 버림)
        x = x[:, 1:, :] 
        
        # 4. 글자 예측
        x = self.head(x)
        
        return x