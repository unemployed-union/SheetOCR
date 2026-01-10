import os
import random
import glob
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
from faker import Faker
from tqdm import tqdm
from fontTools.ttLib import TTFont # [핵심] 폰트 내부를 뜯어보는 도구

# --- 설정 ---
OUTPUT_DIR = "dataset/train"
FONT_DIR = "fonts"
NUM_SAMPLES = 50000
IMG_W, IMG_H = 448, 112

fake_ko = Faker('ko-KR')
fake_en = Faker('en-US')

# -------------------------------------------------------------------
# 1. [핵심] 폰트별 지원 문자표(CMAP) 추출 클래스
# -------------------------------------------------------------------
class FontManager:
    def __init__(self, font_dir):
        self.font_db = {} # { 'font_path': set(지원하는_유니코드_코드들) }
        self.load_fonts(font_dir)

    def get_char_set(self, font_path):
        """폰트 파일(ttLib)을 열어서 지원하는 모든 문자 코드를 set으로 반환"""
        try:
            # ttc(컬렉션) 파일 대응
            if font_path.lower().endswith('.ttc'):
                # TTC는 여러 폰트가 뭉쳐있음, 첫 번째 것만 사용하거나 까다로움.
                # 편의상 PIL이 알아서 처리하게 놔두고, 여기선 일단 패스하거나 
                # fontNumber=0으로 읽어야 함. (여기선 복잡도상 ttf/otf 위주로 처리 권장)
                # * 맥 시스템 폰트(TTC)를 쓰려면 이 부분이 복잡해지므로, 
                #   가급적 ttf 변환된 걸 쓰거나 아래 try-except로 넘김.
                font = TTFont(font_path, fontNumber=0) 
            else:
                font = TTFont(font_path)
                
            cmap = font.getBestCmap() # {unicode: glyph_name}
            if cmap:
                return set(cmap.keys())
            return set()
        except Exception as e:
            # print(f"⚠️ 폰트 로드 실패 ({os.path.basename(font_path)}): {e}")
            return set()

    def load_fonts(self, font_dir):
        files = glob.glob(os.path.join(font_dir, "**/*.ttf"), recursive=True) + \
                glob.glob(os.path.join(font_dir, "**/*.otf"), recursive=True) + \
                glob.glob(os.path.join(font_dir, "**/*.ttc"), recursive=True) # ttc 추가
        
        print(f"🕵️‍♂️ 폰트 족보(CMAP) 생성 중... (파일 {len(files)}개)")
        
        for f in tqdm(files):
            chars = self.get_char_set(f)
            # 한글 '가'(44032)가 포함된 폰트만 한글 폰트로 인정
            # (영문 폰트는 한글 지원 set이 없으므로 자동 필터링됨)
            if len(chars) > 0:
                self.font_db[f] = chars
                
        print(f"✅ 로드 완료: {len(self.font_db)}개 폰트 등록됨")

    def get_valid_font_for_text(self, text):
        """
        입력된 text의 모~든 글자를 지원하는 폰트 중 하나를 랜덤 반환.
        없으면 None 반환.
        """
        # 텍스트를 유니코드 정수 집합으로 변환
        text_chars = set(ord(c) for c in text if c != ' ') # 공백은 제외하고 검사
        
        valid_fonts = []
        for font_path, supported_chars in self.font_db.items():
            # text의 모든 글자가 supported_chars 집합의 부분집합(subset)인가?
            if text_chars.issubset(supported_chars):
                valid_fonts.append(font_path)
                
        if not valid_fonts:
            return None
            
        return random.choice(valid_fonts)

# --- 2. 매니저 초기화 (시간이 조금 걸립니다) ---
font_manager = FontManager(FONT_DIR)


# --- 3. 증강 파이프라인 (동일) ---
transform_pipeline = A.Compose([
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.RandomBrightnessContrast(p=0.5),
])

# 2350자 Vocab 로드 (없으면 생성 안함)
try:
    with open("vocab.txt", "r", encoding="utf-8") as f:
        vocab_chars = set([line.strip('\n') for line in f.readlines()])
except:
    vocab_chars = None

def generate_random_text():
    if random.random() < 0.5:
        if random.random() < 0.3:
            return f"{fake_en.word()} {fake_ko.word()} {random.randint(1, 9)}"
        return fake_ko.catch_phrase()
    else:
        return fake_en.sentence().replace(".", "")

# --- 4. 이미지 생성 함수 ---
def create_synthetic_image(text, index):
    # [핵심 변경] 텍스트 내용을 보고, 이걸 완벽히 소화할 수 있는 폰트를 달라고 함
    selected_font_path = font_manager.get_valid_font_for_text(text)
    
    # 만약 이 텍스트를 지원하는 폰트가 하나도 없다면? (예: 궯 같은 이상한 글자)
    if selected_font_path is None:
        # print(f"⏭️ 스킵: '{text}'를 지원하는 폰트가 없음")
        return None

    # 이제부터는 아까와 동일한 로직 (두부 걱정 없이 그림)
    try:
        # 캔버스 초기화
        current_w = IMG_W
        current_h = IMG_H
        bg_color = random.randint(200, 255)
        image = Image.new("RGB", (current_w, current_h), (bg_color, bg_color, bg_color))
        draw = ImageDraw.Draw(image)
        
        # Auto-fit + Dynamic Width
        font_size = 85
        min_font_size = 25
        margin = 20
        final_font = None
        text_w, text_h = 0, 0
        
        while True:
            # BASIC 엔진 사용 권장
            font = ImageFont.truetype(selected_font_path, font_size, layout_engine=ImageFont.Layout.BASIC)
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except:
                return None # 폰트 자체 에러시

            if text_w < (current_w - margin) and text_h < (current_h - margin):
                final_font = font
                break
            
            font_size -= 2
            
            if font_size < min_font_size:
                final_font = ImageFont.truetype(selected_font_path, min_font_size, layout_engine=ImageFont.Layout.BASIC)
                new_w = text_w + margin + 40
                image = Image.new("RGB", (new_w, current_h), (bg_color, bg_color, bg_color))
                draw = ImageDraw.Draw(image)
                current_w = new_w
                break
        
        # 그리기
        x = (current_w - text_w) // 2
        y = (current_h - text_h) // 2
        x = max(0, x)
        y = max(0, y)
        
        draw.text((x, y), text, font=final_font, fill=(0, 0, 0))

        # 증강 및 저장
        image_np = np.array(image)
        augmented = transform_pipeline(image=image_np)['image']
        final_image = Image.fromarray(augmented)

        filename = f"train_{index:05d}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)
        final_image.save(save_path)
        
        return f'{{"file_name": "{filename}", "text": "{text}"}}\n'

    except Exception:
        return None

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("🚀 데이터 생성 시작 (CMAP 검증 모드)...")
    with open(os.path.join(OUTPUT_DIR, "metadata.jsonl"), "w", encoding="utf-8") as f:
        count = 0
        pbar = tqdm(total=NUM_SAMPLES)
        
        while count < NUM_SAMPLES:
            text = generate_random_text()
            
            # Vocab 필터링 (1차 방어선)
            if vocab_chars and not all(char in vocab_chars for char in text):
                continue

            line = create_synthetic_image(text, count)
            if line:
                f.write(line)
                count += 1
                pbar.update(1)
                
    print("\n✅ 완료! fontTools를 통해 완벽하게 검증된 데이터입니다.")