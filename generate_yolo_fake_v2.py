import os
import random
import glob
from PIL import Image, ImageDraw, ImageFont

# 설정
OUTPUT_DIR_IMG = "dataset_yolo/train/images"
OUTPUT_DIR_LBL = "dataset_yolo/train/labels"
FONT_DIR = "fonts" # 폰트 폴더 경로 확인하세요!
NUM_SAMPLES = 100 

os.makedirs(OUTPUT_DIR_IMG, exist_ok=True)
os.makedirs(OUTPUT_DIR_LBL, exist_ok=True)

font_paths = glob.glob(os.path.join(FONT_DIR, "*.ttf")) + glob.glob(os.path.join(FONT_DIR, "*.otf"))

def create_fake_sheet_v2(idx):
    # 1. 배경 생성
    w, h = 640, 640
    color = random.randint(230, 255)
    img = Image.new('RGB', (w, h), (color, color, color))
    draw = ImageDraw.Draw(img)

    # 폰트 로드용 헬퍼 함수
    def get_font(size):
        if not font_paths: return ImageFont.load_default()
        try:
            return ImageFont.truetype(random.choice(font_paths), size)
        except:
            return ImageFont.load_default()

    # 2. [추가] 가짜 악보 내용물 채우기 (노이즈)
    # YOLO에게 "이건 제목이 아니야!"라고 알려줄 방해꾼들입니다.
    
    # (A) 오선지 그리기
    for i in range(4, 20): # 위쪽 여백 좀 남기고 시작
        y = i * 30
        draw.line([(0, y), (w, y)], fill=(0, 0, 0), width=1)
        
        # (B) 가짜 음표 (그냥 까만 동그라미/타원)
        if i % 3 != 0: # 띄엄띄엄
            for _ in range(random.randint(5, 15)):
                nx = random.randint(10, w-10)
                ny = y + random.randint(-10, 10)
                # 음표 머리처럼 생긴 타원 그리기
                draw.ellipse((nx, ny, nx+10, ny+8), fill='black')
                # 음표 기둥 (세로선)
                draw.line([(nx+10, ny+4), (nx+10, ny-25)], fill='black', width=1)

    # (C) 가짜 가사 (작은 글씨들)
    # 제목보다 훨씬 작게, 여기저기 뿌림
    for _ in range(10):
        lx = random.randint(10, w-100)
        ly = random.randint(100, h-50) # 제목 위치(상단) 피해서 아래쪽에
        l_text = "lyrics noise sample"
        l_font = get_font(random.randint(10, 15)) # 아주 작은 폰트
        draw.text((lx, ly), l_text, font=l_font, fill='black')

    # 3. 진짜 제목 박기 (주인공)
    text = f"Title {idx} Song"
    t_font_size = random.randint(35, 65) # 가사보다 훨씬 큼!
    t_font = get_font(t_font_size)
    
    # 텍스트 크기 계산
    bbox = draw.textbbox((0, 0), text, font=t_font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # 위치: 상단 중앙 (하지만 약간씩 틀어짐)
    x = (w - text_w) // 2 + random.randint(-30, 30)
    y = random.randint(30, 80) # 상단 고정

    draw.text((x, y), text, font=t_font, fill='black')

    # 4. 라벨 생성
    center_x = (x + text_w / 2) / w
    center_y = (y + text_h / 2) / h
    norm_w = (text_w * 1.1) / w
    norm_h = (text_h * 1.2) / h

    label_str = f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}"

    # 5. 저장
    filename = f"fake_sheet_v2_{idx}"
    img.save(os.path.join(OUTPUT_DIR_IMG, f"{filename}.jpg"))
    with open(os.path.join(OUTPUT_DIR_LBL, f"{filename}.txt"), "w") as f:
        f.write(label_str)

print("🏭 업그레이드된 가짜 데이터(노이즈 포함) 생성 중...")
for i in range(NUM_SAMPLES):
    create_fake_sheet_v2(i)
print("✅ 생성 완료! 이제 YOLO는 제목과 가사를 구별할 수 있게 됩니다.")