# train_rl.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import pandas as pd
from tqdm import tqdm

# 기존 모듈 가져오기
from .dataset import SheetMusicDataset, collate_fn
from .tokenizer import Tokenizer
from .vis_transformer import SimpleViTForOCR

import Levenshtein


def compute_reward(pred_text, target_text):
    """
    Levenshtein 거리 기반 보상 함수
    - 1.0: 완벽하게 일치
    - 0.0: 하나도 안 맞음
    """
    if len(target_text) == 0:
        return 0.0
    
    # 편집 거리 계산 (몇 글자를 고쳐야 정답이 되는지)
    distance = Levenshtein.distance(pred_text, target_text)
    max_len = max(len(pred_text), len(target_text))
    
    # 점수화 (0 ~ 1 사이로 정규화)
    # 거리가 0이면 score는 1.0 (최고)
    score = 1.0 - (distance / max_len)
    
    return score


def train_rl_epoch(model, dataloader, optimizer, device, tokenizer):
    model.train()
    total_reward = 0
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="🚀 RL Training")
    
    for images, targets, target_lengths in progress_bar:
        images = images.to(device)
        batch_size = images.size(0)
        
        # 1. 모델 예측 (Logits 추출)
        # logits shape: [Batch, SeqLen, Vocab]
        logits = model(images)
        
        # -------------------------------------------------------
        # [핵심] SCST (Self-Critical Sequence Training) 알고리즘
        # -------------------------------------------------------
        
        # A. 확률 분포 만들기
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        # B. 두 가지 버전으로 문장 생성
        # (1) Sampling: 확률에 따라 무작위로 뽑아봄 (모험)
        sample_ids = dist.sample() 
        
        # (2) Greedy: 확률이 제일 높은 것만 뽑음 (Baseline/기준점)
        with torch.no_grad():
            greedy_ids = probs.argmax(dim=-1)
            
        # C. 보상(Reward) 계산
        # 배치 내의 각 샘플마다 보상 계산
        rl_loss = 0
        batch_avg_reward = 0
        
        for i in range(batch_size):
            # 텍스트로 디코딩
            sample_seq = sample_ids[i].tolist()
            greedy_seq = greedy_ids[i].tolist()
            
            # 정답 텍스트 가져오기
            # targets는 1차원으로 펴져 있으므로 잘라내야 함 (collate_fn 구조상 복잡해서 idx_to_char로 직접 변환 추천)
            # 여기서는 편의상 dataloader가 target_text를 주면 좋지만, 없으므로 targets 텐서에서 복원
            start = sum(target_lengths[:i])
            end = start + target_lengths[i]
            target_seq = targets[start:end].tolist()
            
            pred_text_sample = tokenizer.decode(sample_seq)
            pred_text_greedy = tokenizer.decode(greedy_seq)
            target_text = "".join([tokenizer.idx_to_char[idx] for idx in target_seq])
            
            # 점수 매기기
            reward_sample = compute_reward(pred_text_sample, target_text)
            reward_greedy = compute_reward(pred_text_greedy, target_text)
            
            # [중요] Advantage (이득) 계산
            # 내가 모험(Sample)을 해서 기준점(Greedy)보다 얼마나 잘했나?
            advantage = reward_sample - reward_greedy
            
            # D. Loss 계산 (Policy Gradient)
            # Log Probability * Advantage
            # 잘했으면(Adv > 0) 그 행동의 확률을 높이고, 못했으면(Adv < 0) 낮춤
            log_prob = dist.log_prob(sample_ids[i]).sum()
            rl_loss -= log_prob * advantage  # Gradient Descent를 위해 (-) 붙임
            
            batch_avg_reward += reward_sample

        # 2. 역전파 (배치 평균)
        optimizer.zero_grad()
        (rl_loss / batch_size).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += rl_loss.item()
        total_reward += batch_avg_reward / batch_size
        
        progress_bar.set_postfix({
            "Loss": f"{total_loss / (progress_bar.n + 1):.4f}", 
            "Reward": f"{total_reward / (progress_bar.n + 1):.4f}" # 1.0에 가까울수록 좋음
        })

    return total_loss / len(dataloader), total_reward / len(dataloader)

def main():
    # --- 설정 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32 # RL은 메모리 많이 먹으니 배치를 좀 줄이세요
    LR = 5e-6       # [중요] RL은 학습률을 아주아주 낮게 잡아야 합니다! (Supervised의 1/100)
    EPOCHS = 20
    
    # 1. Vocab & Tokenizer
    with open("vocab.txt", "r", encoding="utf-8") as f:
        vocab_list = [line.replace('\n', '').replace('\r', '') for line in f.readlines()]
    tokenizer = Tokenizer(vocab_list)
    
    # 2. Dataset
    df = pd.read_json("dataset_vit/train_final/metadata.jsonl", lines=True)

    rl_transform = transforms.Compose([
        transforms.Resize((112, 448)),        # 텐서로 변환 (0~1)
        transforms.Normalize(mean=[0.5], std=[0.5]) # [핵심] -1~1로 정규화
    ])

    # RL 할 때는 Augmentation을 끄거나 약하게 하는 게 좋습니다 (정답을 확실히 맞추는 게 목표)
    dataset = SheetMusicDataset("dataset_vit/train_final", df, tokenizer, transform=rl_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # 3. 모델 로드
    model = SimpleViTForOCR(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=384, num_layers=6 # 기존 설정 유지
    ).to(DEVICE)
    
    # [필수] 지도 학습(Supervised)으로 똑똑해진 모델 불러오기
    # 70% 정확도 찍은 그 파일 경로를 넣으세요!
    pretrained_path = "best_model_one.pth" # 파일명 확인 필수!
    print(f"🔄 Pretrained Model 로드 시도: {pretrained_path}")
    
    # 1. 일단 불러오기
    state_dict = torch.load(pretrained_path, map_location=DEVICE)
    
    # 2. 'module.' 접두사 제거 (DataParallel로 저장된 경우 대비)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") # module. 제거
        new_state_dict[name] = v
        
    # 3. 모델에 넣기 (strict=True로 변경해서 안 맞으면 에러 나게 함!)
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("✅ 모델 가중치가 완벽하게 로드되었습니다!")
    except Exception as e:
        print(f"🚨 [치명적 오류] 가중치 로드 실패! 모델 구조가 다릅니다.")
        print(f"에러 메시지: {e}")
        # 여기서 에러가 나면, main.py의 모델 설정(층수, 히든사이즈 등)과 
        # train_rl.py의 모델 설정이 똑같은지 확인해야 합니다.
        exit()
    
    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    print("🔥 강화 학습(RL) 시작! (목표: Reward 1.0)")
    
    for epoch in range(EPOCHS):
        loss, reward = train_rl_epoch(model, dataloader, optimizer, DEVICE, tokenizer)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] RL Loss: {loss:.4f} | Avg Reward: {reward:.4f}")
        
        # 보상이 높을 때 저장
        if reward > 0.90:
            torch.save(model.state_dict(), f"rl_model_epoch_{epoch+1}_rew_{reward:.2f}.pth")

if __name__ == "__main__":
    main()