import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

# --- [설정] ---
# 원본 이미지가 있는 폴더
ORIGINAL_DIR = "dataset/train"       

# 변환된 이미지를 저장할 폴더 (이 폴더가 자동으로 생성됩니다)
NEW_DIR = "dataset/train_resized"    

# 목표 크기 (가로 448, 세로 112)
TARGET_SIZE = (448, 112)             

# 메타데이터 파일 경로
METADATA_FILE = "dataset/train/metadata.jsonl"     

def resize_and_convert_all():
    # 1. 저장할 폴더가 없으면 만듭니다.
    if not os.path.exists(NEW_DIR):
        os.makedirs(NEW_DIR)
        print(f"📁 저장 폴더 생성 완료: {NEW_DIR}")

    # 2. 메타데이터 읽기
    print("📂 메타데이터 목록을 읽는 중...")
    if not os.path.exists(METADATA_FILE):
        print(f"🚨 에러: {METADATA_FILE} 파일이 없습니다! 경로를 확인해주세요.")
        return

    df = pd.read_json(METADATA_FILE, lines=True)
    
    print(f"🚀 총 {len(df)}장의 이미지를 '흑백 + 리사이즈' 변환합니다...")

    # 3. 하나씩 변환해서 저장
    for idx in tqdm(range(len(df))):
        file_name = df.iloc[idx]['file_name']
        src_path = os.path.join(ORIGINAL_DIR, file_name)
        dst_path = os.path.join(NEW_DIR, file_name)

        try:
            with Image.open(src_path) as img:
                # [핵심] L = Grayscale (흑백), LANCZOS = 고품질 리사이징
                img = img.convert("L").resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                
                # 저장 (품질 95로 설정하여 화질 저하 최소화)
                img.save(dst_path, quality=95)
                
        except Exception as e:
            print(f"❌ 변환 실패 ({file_name}): {e}")

    print("\n✅ 전처리 완료!")
    print(f"👉 '{NEW_DIR}' 폴더에 흑백 이미지가 저장되었습니다.")

if __name__ == "__main__":
    resize_and_convert_all()