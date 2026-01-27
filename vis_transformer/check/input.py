import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from ..dataset import SheetMusicDataset, collate_fn
from ..tokenizer import Tokenizer
import sys

def check_data_text_only():
    print("🕵️‍♂️ WSL용 데이터 정밀 진단 시작...\n")

    # 1. Vocab 점검 (스페이스바 생존 확인)
    print("[1] Vocab 파일 점검")
    try:
        with open("vocab.txt", "r", encoding="utf-8") as f:
            # [핵심] strip() 대신 replace로 읽기
            vocab_list = [line.replace('\n', '').replace('\r', '') for line in f.readlines()]
        
        if ' ' in vocab_list:
            print(f"   ✅ 합격! 스페이스바가 {vocab_list.index(' ')}번 인덱스에 존재합니다.")
        else:
            print("   🚨 불합격! 스페이스바가 없습니다. main.py의 vocab 읽는 부분을 고치세요!")
    except Exception as e:
        print(f"   ❌ 에러: {e}")
        return

    # 2. 토크나이저 점검
    tokenizer = Tokenizer(vocab_list)
    print(f"\n[2] 토크나이저 테스트")
    test_str = "434 곡" # 숫자와 공백 포함
    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded)
    print(f"   입력: '{test_str}'")
    print(f"   변환(ID): {encoded}")
    print(f"   복원: '{decoded}'")
    
    if len(encoded) < 4: # 4,3,4, ,곡 (5개)여야 하는데 줄었다면
        print("   🚨 경고: 글자 수가 줄어들었습니다! (삭제됨)")
    else:
        print("   ✅ 정상: 모든 글자가 잘 살아있습니다.")

    # 3. 이미지 데이터(픽셀) 점검
    print(f"\n[3] 이미지 텐서 값 점검 (눈 대신 숫자로 확인)")
    try:
        df = pd.read_json("dataset_vit/train_final/metadata.jsonl", lines=True).iloc[:4]
        # Transform 없이 Raw 데이터 확인 (dataset 내부 로직만 통과)
        dataset = SheetMusicDataset("dataset_vit/train_final", df, tokenizer, transform=None)
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        
        images, targets, lengths = next(iter(loader))
        
        # 통계 계산
        min_val = images.min().item()
        max_val = images.max().item()
        mean_val = images.mean().item()
        
        print(f"   📊 픽셀 범위: {min_val:.4f} ~ {max_val:.4f}")
        print(f"   📊 픽셀 평균: {mean_val:.4f}")

        # 판단 로직
        if min_val == max_val:
            print("   🚨 [치명적] 이미지가 단색(전부 검정 or 흰색)입니다! 전처리 코드를 확인하세요.")
        elif min_val < 0 and max_val > 0:
            print("   ✅ 정상: 이미지가 -1 ~ 1 사이로 잘 정규화되어 있습니다.")
        elif min_val >= 0 and max_val <= 1.0:
             print("   ✅ 정상: 이미지가 0 ~ 1 사이로 로드되었습니다.")
        else:
            print("   ⚠️ 주의: 픽셀 범위가 특이합니다. (하지만 단색은 아님)")

    except Exception as e:
        print(f"   ❌ 데이터 로드 중 에러: {e}")

if __name__ == "__main__":
    check_data_text_only()