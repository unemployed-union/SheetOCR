import os
import random
import glob
import json
from PIL import Image, ImageDraw, ImageFont
from faker import Faker

# -------------------------------------------------
# ⚙️ 설정 (Configuration)
# -------------------------------------------------
OUT_DIR = "dataset_vit/train_final"  # 저장 경로
NUM_SAMPLES = 20000                  # 생성할 이미지 수
IMG_W, IMG_H = 448, 112              # ViT 입력 크기 (고정)

# 폰트 경로 (사용자가 직접 폴더를 만들고 폰트를 넣어야 함)
FONT_THIN_DIR = "fonts/thin"  # 본문용 (명조, 나눔고딕 등)
FONT_BOLD_DIR = "fonts/bold"  # 제목용 (배민주아, 격동고딕, G마켓산스Bold 등)

# Faker 라이브러리 초기화
fake_ko = Faker('ko_KR')
fake_en = Faker('en_US')

# 폴더 생성
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------
# 📖 성경/찬송가 스타일 생성기 (Custom Generator)
# -------------------------------------------------
def get_hymn_style_text(count=2):
    """성경이나 찬송가에 자주 나오는 단어들을 조합"""
    vocab = [
        "사랑", "은혜", "주님", "믿음", "소망", "구원", "영광", "찬양", "기도", 
        "예배", "축복", "평화", "기쁨", "감사", "말씀", "진리", "생명", "하늘",
        "거룩", "능력", "지혜", "위로", "인도", "보혈", "십자가", "부활", "천국",
        "나의", "예수", "하나님", "성령", "임재", "약속", "선한", "목자"
    ]
    # 랜덤으로 2~3개 단어를 뽑아서 이어 붙임 (예: "거룩한 말씀", "나의 사랑")
    return " ".join(random.sample(vocab, k=count))

# -------------------------------------------------
# 🧠 지능형 텍스트 생성기 (Smart Text Generator)
# -------------------------------------------------
def generate_smart_text(is_title_mode):
    """
    모드에 따라 확률적으로 다양한 스타일의 텍스트를 생성
    """
    
    # =========================================================
    # (A) 제목 모드 (Bold 폰트 사용) - 짧고 굵은 글씨
    # =========================================================
    if is_title_mode:
        rand = random.random()
        
        # 1. [숫자/찬송가형] (30%) -> "찬송가 28장", "No. 1"
        if rand < 0.3:
            number = random.randint(1, 600)
            if random.random() < 0.4:
                prefix = random.choice(["찬송가", "장", "제", "곡"])
                return f"{prefix} {number}"
            elif random.random() < 0.7:
                suffix = random.choice(["장", "과", "번", "곡", "장 찬양"])
                return f"{number}{suffix}"
            else:
                prefix = random.choice(["Hymn", "No.", "Psalm", "Chapter"])
                return f"{prefix} {number}"

        # 2. [성경 문구형] (25%) -> "주님 사랑", "은혜의 강" (우리가 만든 생성기)
        elif rand < 0.55:
            return get_hymn_style_text(count=random.randint(2, 3))

        # 3. [일반/비즈니스형] (15%) -> "창의적인 생각" (Faker 활용)
        elif rand < 0.70:
            # Faker의 슬로건에서 앞 3단어만 가져옴
            text = fake_ko.catch_phrase() if random.random() < 0.5 else fake_ko.bs()
            return " ".join(text.split(" ")[:3])

        # 4. [영어/혼합형] (30%) -> "Amazing Grace", "Jesus 사랑"
        else:
            if random.random() < 0.6:
                # [영어 제목] 2~6단어의 영어 문구 (Body tough agent... 대응)
                sentence = fake_en.sentence().replace(".", "")
                word_count = random.randint(2, 6) 
                return " ".join(sentence.split(" ")[:word_count]).title()
            else:
                # [한영 혼용] "My 주님"
                return f"{fake_en.word().capitalize()} {get_hymn_style_text(1)}"

    # =========================================================
    # (B) 본문 모드 (Thin 폰트 사용) - 길고 얇은 글씨
    # =========================================================
    else:
        rand = random.random()
        
        # 1. [성경 긴 문장] (35%) -> "사랑 은혜 주님..."
        if rand < 0.35:
            return get_hymn_style_text(count=random.randint(5, 8))
            
        # 2. [Faker 일반 한글 문장] (30%) -> "이 제품은..." (비즈니스/슬로건 조합)
        elif rand < 0.65:
            return fake_ko.catch_phrase() + " " + fake_ko.bs()
            
        # 3. [영어 긴 문장] (35%) -> "Lorem ipsum..." (원래 데이터 스타일)
        else:
            text = fake_en.sentence().replace(".", "")
            # 가끔 끝에 숫자 붙이기 (가사 절 번호 흉내)
            if random.random() < 0.1: text += f" {random.randint(1, 9)}"
            return text

# -------------------------------------------------
# 🎨 이미지 생성 함수 (Image Generator)
# -------------------------------------------------
def create_data(idx, thin_fonts, bold_fonts):
    # 50% 확률로 [제목 스타일] vs [본문 스타일] 결정
    is_title_style = random.random() < 0.5
    
    if is_title_style:
        # [제목 스타일] 배경 흰색, 폰트 굵게, 글씨 크게
        bg_color = 255 
        font_list = bold_fonts if bold_fonts else thin_fonts
        font_size = random.randint(55, 85)
        text = str(generate_smart_text(True))
    else:
        # [본문 스타일] 배경 노이즈, 폰트 얇게, 글씨 작게
        bg_color = random.randint(230, 255)
        font_list = thin_fonts if thin_fonts else bold_fonts
        font_size = random.randint(30, 50)
        text = str(generate_smart_text(False))

    # 1. 캔버스 생성 (Grayscale)
    img = Image.new('L', (IMG_W, IMG_H), bg_color)
    draw = ImageDraw.Draw(img)

    # 2. 폰트 로드
    font_path = random.choice(font_list) if font_list else None
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    # 3. 텍스트 크기 계산 & 중앙 정렬
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # [안전장치] 글자가 이미지보다 길면 다시 생성 (재귀 호출)
    if text_w > IMG_W - 20: 
        return create_data(idx, thin_fonts, bold_fonts)

    x = (IMG_W - text_w) // 2
    y = (IMG_H - text_h) // 2
    draw.text((x, y), text, font=font, fill=0)
    
    # 4. 노이즈 추가 (본문 스타일일 때만)
    if not is_title_style:
        for _ in range(random.randint(100, 400)):
            draw.point((random.randint(0, IMG_W), random.randint(0, IMG_H)), 
                       fill=random.randint(150, 200))

    # 5. 저장
    file_name = f"train_{idx:05d}.jpg"
    img.save(os.path.join(OUT_DIR, file_name))
    
    return file_name, text

# -------------------------------------------------
# 🚀 메인 실행 블록
# -------------------------------------------------
if __name__ == "__main__":
    # 폰트 로드
    thin_fonts = glob.glob(os.path.join(FONT_THIN_DIR, "*.ttf")) + glob.glob(os.path.join(FONT_THIN_DIR, "*.otf"))
    bold_fonts = glob.glob(os.path.join(FONT_BOLD_DIR, "*.ttf")) + glob.glob(os.path.join(FONT_BOLD_DIR, "*.otf"))

    print(f"📂 얇은 폰트(Thin): {len(thin_fonts)}개")
    print(f"📂 굵은 폰트(Bold): {len(bold_fonts)}개")

    if not thin_fonts and not bold_fonts:
        print("❌ [오류] 폰트 파일이 없습니다!")
        print(f"👉 '{FONT_THIN_DIR}'과 '{FONT_BOLD_DIR}' 폴더에 폰트를 넣어주세요.")
    else:
        print(f"🚀 [최종완성] 하이브리드 데이터 생성 시작... ({NUM_SAMPLES}장)")
        
        jsonl_path = os.path.join(OUT_DIR, "metadata.jsonl")
        
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i in range(NUM_SAMPLES):
                fname, label = create_data(i, thin_fonts, bold_fonts)
                
                # JSONL 포맷 저장
                line = {"file_name": fname, "text": label}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                
                if (i + 1) % 2000 == 0:
                    print(f"   ... {i + 1}장 생성 완료")
                
        print("-" * 50)
        print("✅ 모든 작업 완료!")
        print(f"📁 데이터 폴더: {OUT_DIR}")
        print(f"📄 라벨 파일: {jsonl_path}")
        print("이제 이 데이터로 학습하면 '은혜', '찬송가 28장', 'English Title' 모두 인식합니다!")