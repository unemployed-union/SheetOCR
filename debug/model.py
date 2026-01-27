import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- [1] 모델 정의 (우리가 만든 Hybrid 구조) ---
class HybridEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768):
        super().__init__()
        # CNN Backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),  # /2
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),           # /4
            nn.BatchNorm2d(64), nn.ReLU(),
            # 높이만 줄이고 너비(시간)는 유지
            nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), 
            nn.BatchNorm2d(128), nn.ReLU(),
        )
        # 128채널 * 14높이 = 1792
        self.proj = nn.Linear(128 * 14, embed_dim)

    def forward(self, x):
        # x: (Batch, 3, 112, 448)
        x = self.cnn(x)  # -> (Batch, 128, 14, 112) [C, H, W]
        
        # [핵심 디버깅] 차원 확인
        # 위에는 H, 아래는 글자 머리. 이 정보를 세로로 합침
        # (Batch, C, H, W) -> (Batch, W, C, H) -> (Batch, W, C*H)
        x = x.permute(0, 3, 1, 2) 
        x = x.flatten(2) # (Batch, 112, 1792)
        x = self.proj(x) # (Batch, 112, 768)
        return x

# [수정된 모델] 트랜스포머 제거, CNN 직결
class SimpleViTForOCR(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 1. HybridEmbed (CNN)
        self.embed = HybridEmbed(embed_dim=768)
        
        # 2. [수정] Transformer 제거! 
        # 복잡한 연산 없이 CNN 출력을 바로 분류기에 넣습니다.
        # self.pos_embed = ... (삭제)
        # self.encoder = ... (삭제)
        
        # 3. Head (분류기)
        self.head = nn.Linear(768, vocab_size)

    def forward(self, x):
        # 입력 -> CNN(HybridEmbed) -> (Batch, 112, 768)
        x = self.embed(x)
        
        # Transformer 없이 바로 예측
        # x = x + self.pos_embed (삭제)
        # x = self.encoder(x) (삭제)
        
        return self.head(x)

# --- [2] 가짜 데이터 생성 (고정된 패턴) ---
# 배치 2개, 이미지 크기 112x448
dummy_images = torch.randn(2, 3, 112, 448) 

# 정답: 1번은 "ABC"(1,2,3), 2번은 "A"(1) 라고 가정
# Target Lengths: 첫번째는 3글자, 두번째는 1글자
target_lengths = torch.tensor([3, 1], dtype=torch.long)
targets = torch.tensor([1, 2, 3, 1], dtype=torch.long) # 다 이어붙임

# --- [3] 학습 루프 (오버핏 테스트) ---
# Vocab Size: 0(Blank) + 1,2,3(글자) = 4개
model = SimpleViTForOCR(vocab_size=5) 
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)

print("🚀 산소호흡기 테스트 시작...")
model.train()

for epoch in range(50):
    optimizer.zero_grad()
    
    outputs = model(dummy_images) # (Batch, 112, Vocab)
    
    # CTC Loss 입력 형태: (Time, Batch, Vocab)
    outputs = outputs.permute(1, 0, 2)
    log_probs = nn.functional.log_softmax(outputs, dim=2)
    
    # Input Lengths: 모델이 뱉은 시간 길이 (112)
    input_lengths = torch.full(size=(2,), fill_value=112, dtype=torch.long)
    
    loss = criterion(log_probs, targets, input_lengths, target_lengths)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# --- [4] 예측 확인 ---
print("\n[결과 확인]")
with torch.no_grad():
    model.eval()
    outputs = model(dummy_images)
    pred = outputs.argmax(dim=2) # (Batch, 112)
    print(f"예측값(인덱스) 0번 샘플 앞부분: {pred[0, :10].tolist()}")
    # 정답 1, 2, 3이 보여야 함 (중간에 0이 섞여 있어도 됨)