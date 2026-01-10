import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    """
    1. 이미지를 패치로 자르고 (Conv2d)
    2. 위치 정보를 더해주는 (Sinusoidal PE) 역할
    """
    def __init__(self, in_channels=3, patch_size=16, emb_size=384):
        super().__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        
        # (1) 패치 투영: 이미지를 16x16 조각으로 잘라 384차원으로 변환
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        
        # (2) [CLS] 토큰: 문장 전체의 의미를 담을 그릇 (학습 가능)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        
        # (주의) 위치 정보(pos_embed)는 학습 파라미터(nn.Parameter)로 만들지 않고
        # forward에서 그때그때 수학 공식으로 만들어서 더할 겁니다. (Dynamic Width 지원용)

    def get_sinusoidal_encoding(self, n_patches, dim, device):
        # 1. CPU에서 먼저 계산 (Mac 친화적)
        position = torch.arange(n_patches, dtype=torch.float, device="cpu").unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device="cpu") * -(math.log(10000.0) / dim))
        
        pe = torch.zeros(n_patches, dim, device="cpu")
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 2. 마지막에 GPU로 전송
        return pe.unsqueeze(0).to(device)

    def forward(self, x):
        # x: (Batch, 3, H, W)
        
        # 1. 패치 자르기
        # (B, 384, H/16, W/16) -> (B, 384, N) -> (B, N, 384)
        x = self.proj(x).flatten(2).transpose(1, 2)
        
        B, N, D = x.shape
        
        # 2. [CLS] 토큰 붙이기
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 
        # 이제 길이는 N + 1 이 됩니다.
        
        # 3. 위치 정보 생성 및 더하기 (여기서 오류 해결!)
        # 현재 길이에 딱 맞는 위치 정보를 즉석에서 만듦
        pos_embed = self.get_sinusoidal_encoding(N + 1, D, x.device)
        x = x + pos_embed
        
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