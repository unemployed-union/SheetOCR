import pandas as pd  # 데이터프레임 로딩용 추가
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torchvision import transforms
from tqdm import tqdm

# 직접 만든 모듈들 import
from .tokenizer import Tokenizer
from .vis_transformer import SimpleViTForOCR  # 직접 짠 커스텀 모델
from .dataset import SheetMusicDataset, collate_fn

# 상단 import 추가
from torch import amp

def train(model, dataloader, criterion, optimizer, device, tokenizer, scheduler=None):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    # [추가] GradScaler는 CUDA용이라 MPS에서는 보통 안 써도 되지만, 
    # PyTorch 최신 버전에서는 MPS도 scaler를 지원하기 시작했습니다. 
    # 안전하게 autocast만 먼저 적용해봅니다.

    for images, targets, target_lengths in progress_bar:
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()

        # [핵심] Autocast 적용 (MPS 모드)
        # 연산을 Float16으로 압축해서 수행합니다.
        with amp.autocast(device_type="mps", dtype=torch.float16):
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)
            log_probs = nn.functional.log_softmax(outputs, dim=2)
            input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long).to(device)
            
            loss = criterion(log_probs.cpu(), targets.cpu(), input_lengths.cpu(), target_lengths.cpu())

        # 역전파
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"Loss": loss.item()})

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, tokenizer):
    model.eval()
    total_loss = 0
    correct_count = 0
    total_count = 0
    sample_count = 0

    with torch.no_grad():
        for images, targets, target_lengths in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            outputs = model(images)

            # Loss 계산용
            outputs_loss = outputs.permute(1, 0, 2)
            log_probs = nn.functional.log_softmax(outputs_loss, dim=2)
            input_lengths = torch.full(
                size=(images.size(0),), fill_value=outputs.size(1), dtype=torch.long
            ).to(device)

            loss = criterion(log_probs.cpu(), targets.cpu(),
                             input_lengths.cpu(), target_lengths.cpu())
            total_loss += loss.item()

            # 정확도 계산용 (Greedy Decoding)
            pred_indices = outputs.argmax(dim=2)

            current_target_idx = 0
            batch_size = images.size(0)

            for i in range(batch_size):
                # 예측값 문자열로 변환
                pred_seq = pred_indices[i].tolist()
                # 중복 제거 로직은 tokenizer 안에 있다고 가정
                pred_text = tokenizer.decode(pred_seq)

                # 정답 문자열로 변환
                t_len = target_lengths[i].item()
                target_seq = targets[current_target_idx:
                                     current_target_idx + t_len].tolist()
                current_target_idx += t_len

                # 정답은 단순 리스트 변환 (idx_to_char 이용)
                target_text = "".join([tokenizer.idx_to_char[idx]
                                      for idx in target_seq])

                if pred_text == target_text:
                    correct_count += 1
                total_count += 1

                if sample_count < 2:  # 에폭당 2개만 샘플 출력
                    print(
                        f"   [검증] 정답: {target_text[:20]:<20} | 예측: {pred_text[:20]}")
                    sample_count += 1

    avg_loss = total_loss / len(dataloader)
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0.0

    return avg_loss, accuracy


def main():
    # --- [설정] ---
    BATCH_SIZE = 32        # RAM 캐싱했으니 64도 거뜬함 (안되면 32로 줄이기)
    LEARNING_RATE = 5e-4   # 1e-4 -> 2e-4 (배치 늘렸으니 조금 올림)
    EPOCHS = 80           # 넉넉하게 잡고 Early Stopping 하세요
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        DEVICE = "mps"  # Mac용

    # 1. Tokenizer 로드
    vocab_list = []
    with open("vocab.txt", "r", encoding="utf-8") as f:
        vocab_list = [line.strip('\n') for line in f.readlines()]
    tokenizer = Tokenizer(vocab_list)

    # 2. 데이터셋 준비 (Pandas로 먼저 읽기)
    transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # JSONL 파일을 읽어서 DataFrame으로 만듭니다.
    print("📂 메타데이터 로딩 중...")
    df = pd.read_json("dataset/train/metadata.jsonl", lines=True)

    # Dataset 생성 (여기서 RAM 캐싱이 일어남 - 시간 좀 걸림)
    full_dataset = SheetMusicDataset(
        root_dir="dataset/train_resized",
        df=df,
        tokenizer=tokenizer,
        transform=transform
    )

    # Train/Val 분리
    train_size = int(0.9 * len(full_dataset))  # 검증 데이터 10%만
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size])

    # DataLoader (RAM 캐싱을 썼으므로 num_workers는 적어도 됨)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )

    # 3. 모델 초기화 (Custom ViT)
    # 직접 구현한 SimpleViTForOCR 사용
    model = SimpleViTForOCR(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=384,   # ViT Small급
        # num_heads=6,     # 384 / 64 = 6
        # num_layers=6     # 레이어 6개 (공부용으로 적당)
    ).to(DEVICE)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 4. 스케줄러 설정 (verbose 삭제, mode='max' 확인)
    # 정확도(Acc)가 안 오르면 LR을 깎습니다.
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,              # 최대 학습률 (여기까지 찍고 내려옴)
        epochs=EPOCHS,            # 전체 에폭 수
        steps_per_epoch=len(train_loader),
        pct_start=0.1,            # 전체 과정의 앞부분 10% 동안 LR을 올림 (Warm-up)
        anneal_strategy='cos'     # 코사인 곡선으로 부드럽게
    )

    print(f"🔥 학습 시작! (Device: {DEVICE})")

    # --- [학습 루프] ---
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, criterion,
                           optimizer, DEVICE, tokenizer)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, DEVICE, tokenizer)

        # [중요] 스케줄러에게 정확도를 알려줌
        scheduler.step(val_acc)

        # 현재 LR 찍어보기
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.8f}")

        # 모델 저장 (정확도 오를 때만 저장하는 로직 추가하면 더 좋음)
        if val_acc > 80:  # 80% 넘으면 저장 시작
            torch.save(model.state_dict(),
                       f"model_epoch_{epoch+1}_acc_{val_acc:.1f}.pth")

    # 최종 저장
    torch.save(model.state_dict(), "final_model.pth")


if __name__ == "__main__":
    main()
