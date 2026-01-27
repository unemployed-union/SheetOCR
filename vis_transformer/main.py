import pandas as pd  # 데이터프레임 로딩용 추가
import torch
import os
from torch import GradScaler
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


def train(model, dataloader, criterion, optimizer, device, tokenizer, scheduler=None):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    scaler = torch.amp.GradScaler('cuda')  # AMP 사용 (필수)

    # [설정] 실제로는 16개씩 넣지만, 4번 모아서 업데이트하므로 64개 효과
    accumulation_steps = 4

    optimizer.zero_grad()  # 루프 시작 전 초기화

    for idx, (images, targets, target_lengths) in enumerate(progress_bar):
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)
            log_probs = nn.functional.log_softmax(outputs, dim=2)
            input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(
                0), dtype=torch.long).to(device)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            # [핵심 1] Loss를 나누기 (4번 더할 거니까 미리 1/4로 나눔)
            loss = loss / accumulation_steps

        # Backward (기울기 계산만 하고 업데이트는 아직 안 함)
        scaler.scale(loss).backward()

        # [수정] 4번째 배치거나, 혹은 '마지막' 배치라면 업데이트!
        if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 로깅용으로는 다시 곱해서 원래 loss 값을 보여줌
        current_loss = loss.item() * accumulation_steps
        epoch_loss += current_loss
        progress_bar.set_postfix({"Loss": current_loss})

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
    BATCH_SIZE = 64        # RAM 캐싱했으니 64도 거뜬함 (안되면 32로 줄이기)
    LEARNING_RATE = 1e-4   # 1e-4 -> 2e-4 (배치 늘렸으니 조금 올림)
    EPOCHS = 100         # 넉넉하게 잡고 Early Stopping 하세요
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        DEVICE = "mps"  # Mac용

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 1. Tokenizer 로드
    vocab_list = []

        # [수정 코드] 엔터와 리턴만 제거하고, 스페이스바는 살려둡니다.
    with open("vocab.txt", "r", encoding="utf-8") as f:
        vocab_list = [line.replace('\n', '').replace('\r', '')
                    for line in f.readlines()]

    # [확인 사살용 코드 - 실행 시 콘솔에 뜸]
    if ' ' in vocab_list:
        print(f"✅ Vocab 로드 성공! 스페이스바가 {vocab_list.index(' ')}번 인덱스에 있습니다.")
    else:
        print("🚨 비상! 여전히 스페이스바가 Vocab 리스트에 없습니다.")
        
    tokenizer = Tokenizer(vocab_list)

    # 2. 데이터셋 준비 (Pandas로 먼저 읽기)
    # transform = transforms.Normalize(mean=[0.5], std=[0.5])
    train_transform = transforms.Compose([
        # 확률(p)을 0.5 -> 0.3으로 낮춤 (일단 쉬운 거 많이 보고 배우라고)
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.5))
        ], p=0.5),  # 30% 확률로만 흐리게

        # 밝기 변화도 조금 약하게
        transforms.ColorJitter(brightness=0.1, contrast=0.1),

        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=2,              # 회전 각도 줄임 (3 -> 2)
                translate=(0.02, 0.02),  # 이동 범위 줄임
                scale=(0.99, 1.02),
                fill=0
            )
        ], p=0.5),  # 30% 확률로만 비틀기

        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # 검증용: 깨끗하게 정규화만
    val_transform = transforms.Normalize(mean=[0.5], std=[0.5])

    # JSONL 파일을 읽어서 DataFrame으로 만듭니다.
    print("📂 메타데이터 로딩 중...")
    df = pd.read_json("dataset_vit/train_final/metadata.jsonl", lines=True)

    df = df.sample(frac=1).reset_index(drop=True)  # 전체 셔플
    split_idx = int(0.9 * len(df))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    # 학습 데이터셋 (Augmentation 적용!)
    train_dataset = SheetMusicDataset(
        root_dir="dataset_vit/train_final",
        df=train_df,
        tokenizer=tokenizer,
        transform=train_transform  # <-- 여기에 train_transform 적용
    )

    # 검증 데이터셋 (깨끗함)
    val_dataset = SheetMusicDataset(
        root_dir="dataset_vit/train_final",
        df=val_df,
        tokenizer=tokenizer,
        transform=val_transform    # <-- 여기에 val_transform 적용
    )

    # DataLoader (RAM 캐싱을 썼으므로 num_workers는 적어도 됨)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,      # 0 -> 4 (또는 8) 변경! (CPU가 병렬로 데이터 준비)
        pin_memory=True     # False -> True 변경! (GPU 전송 가속)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,      # 여기도 똑같이
        pin_memory=True     # 여기도 똑같이
    )

    # 3. 모델 초기화 (Custom ViT)
    # 직접 구현한 SimpleViTForOCR 사용
    model = SimpleViTForOCR(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=384,   # ViT Small급
        # num_heads=6,     # 384 / 64 = 6
        # num_layers=12     # 레이어 6개 (공부용으로 적당)
    ).to(DEVICE)

    load_path = "final_model.pth" # 잘 됐던 그 파일

    if os.path.exists(load_path):
        print(f"🔥 {load_path} 로드! 70%에서 다시 등반 시작!")
        # strict=True로 해서 확실하게 로드 (구조 안 바꿨으니까요)
        model.load_state_dict(torch.load(load_path), strict=True)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 4. 스케줄러 설정 (verbose 삭제, mode='max' 확인)
    # 정확도(Acc)가 안 오르면 LR을 깎습니다.
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,      # 3은 너무 급함. 5번 정도는 참아주게 변경
        min_lr=1e-6     # [중요] 아무리 깎아도 0.000001 밑으로는 안 내려감!
    )

    print(f"🔥 학습 시작! (Device: {DEVICE})")

    # --- [학습 루프] ---
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, criterion,
                           optimizer, DEVICE, tokenizer, scheduler=scheduler)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, DEVICE, tokenizer)

        # [중요] 스케줄러에게 정확도를 알려줌
        # scheduler.step(val_acc) # OneCycleLR을 사용하므로 주석처리

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

        scheduler.step(val_acc)

    # 최종 저장
    torch.save(model.state_dict(), "final_model.pth")


if __name__ == "__main__":
    main()
