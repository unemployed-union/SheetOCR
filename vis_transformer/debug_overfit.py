import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

# 사용자님의 모듈 import
from .dataset import SheetMusicDataset, collate_fn
from .tokenizer import Tokenizer
from .vis_transformer import SimpleViTForOCR 

def debug_one_batch():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔍 디버깅 시작 (Device: {DEVICE})")

    # 1. 토크나이저 & 데이터 로드
    vocab_list = [line.strip('\n') for line in open("vocab.txt", "r", encoding="utf-8")]
    tokenizer = Tokenizer(vocab_list)
    
    # [체크 1] 0번이 [PAD]인지 확인
    print(f"🆔 Vocab 0번 ID 확인: '{vocab_list[0]}'")
    if vocab_list[0] != "[PAD]":
        print("🚨 [경고] 0번이 [PAD]가 아닙니다! CTC Loss는 0번을 Blank로 씁니다.")

    # 2. 아주 단순한 Transform (증강 끄고 정규화만)
    transform = transforms.Compose([
        transforms.Resize((112, 448)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 3. 데이터 딱 4개만 가져오기
    df = pd.read_json("dataset_vit/train_final/metadata.jsonl", lines=True).iloc[:4]
    dataset = SheetMusicDataset("dataset_vit/train_final", df, tokenizer, transform)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    # 4. 모델 생성 (기존 설정 그대로)
    model = SimpleViTForOCR(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=384, num_heads=6, num_layers=6 # 혹은 12
    ).to(DEVICE)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0) # LR 높게, 규제 끔

    # 5. 데이터 하나만 반복 학습 (Overfitting)
    images, targets, target_lengths = next(iter(loader))
    images = images.to(DEVICE)
    targets = targets.to(DEVICE)
    target_lengths = target_lengths.to(DEVICE)

    # [체크 2] 입력 값 범위 확인
    print(f"📊 입력 이미지 범위: Min={images.min().item():.2f}, Max={images.max().item():.2f}")
    if images.max() > 200:
        print("🚨 [치명적] 이미지가 0~255 값입니다! 0~1 혹은 -1~1로 정규화되어야 합니다.")

    model.train()
    print("\n🚀 학습 시작 (100 Epoch 동안 4개만 외우기)...")
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(images) # [Batch, Seq, Class]
        
        # CTC Loss 계산
        outputs_log_softmax = nn.functional.log_softmax(outputs, dim=2).permute(1, 0, 2)
        input_lengths = torch.full(size=(4,), fill_value=outputs.size(1), dtype=torch.long).to(DEVICE)
        
        loss = criterion(outputs_log_softmax, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            # 예측 결과 디코딩해서 보여주기
            pred_idx = outputs.argmax(dim=2)[0].tolist() # 첫 번째 샘플만
            pred_str = tokenizer.decode(pred_idx)
            
            # 정답 문자열
            target_str = tokenizer.decode(targets[:target_lengths[0]].tolist())
            
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")
            print(f" -> 정답: {target_str}")
            print(f" -> 예측: {pred_str}")
            print("-" * 30)

debug_one_batch()