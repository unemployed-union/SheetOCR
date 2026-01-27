import os
import urllib.request
import ssl
from concurrent.futures import ThreadPoolExecutor

# 1. 저장할 폴더
SAVE_DIR = "fonts"

# 2. 한글 지원 폰트 리스트 (구글 폰트 OFL 저장소 기준)
# OCR 학습을 위해 고딕, 명조, 필기체, 붓글씨 등 다양한 스타일을 포함했습니다.
KOREAN_FONTS = [
    # --- 고딕 계열 (Standard Sans) ---
    "nanumgothic/NanumGothic-Regular.ttf",
    "nanumgothic/NanumGothic-Bold.ttf",
    "nanumgothic/NanumGothic-ExtraBold.ttf",
    "notosanskr/NotoSansKR-Regular.ttf",
    "notosanskr/NotoSansKR-Bold.ttf",
    "notosanskr/NotoSansKR-Black.ttf",
    "gowundodum/GowunDodum-Regular.ttf",
    "nanumgothiccoding/NanumGothicCoding-Regular.ttf",
    "gothica1/GothicA1-Regular.ttf",
    "gothica1/GothicA1-Black.ttf",
    
    # --- 명조 계열 (Serif) ---
    "nanummyeongjo/NanumMyeongjo-Regular.ttf",
    "nanummyeongjo/NanumMyeongjo-Bold.ttf",
    "notoserifkr/NotoSerifKR-Regular.ttf",
    "notoserifkr/NotoSerifKR-Bold.ttf",
    "gowunbatang/GowunBatang-Regular.ttf",
    "gowunbatang/GowunBatang-Bold.ttf",
    "songmyung/SongMyung-Regular.ttf",

    # --- 필기체/장식체 (Handwriting/Display) -> OCR 난이도 올리기 좋음 ---
    "nanumpenscript/NanumPenScript-Regular.ttf",
    "nanumbrushscript/NanumBrush.ttf",
    "jua/Jua-Regular.ttf",
    "dohyeon/DoHyeon-Regular.ttf",
    "yeonsung/YeonSung-Regular.ttf",
    "sunflower/Sunflower-Medium.ttf",
    "gaegu/Gaegu-Regular.ttf",
    "gaegu/Gaegu-Bold.ttf",
    "himelody/HiMelody-Regular.ttf",
    "gamjaflower/GamjaFlower-Regular.ttf",
    "dokdo/Dokdo-Regular.ttf",
    "eastseadokdo/EastSeaDokdo-Regular.ttf",
    "blackhansans/BlackHanSans-Regular.ttf",
    "cutefont/CuteFont-Regular.ttf",
    "kiranghaerang/KirangHaerang-Regular.ttf",
    "singleday/SingleDay-Regular.ttf",
    "stylish/Stylish-Regular.ttf",
    "gugi/Gugi-Regular.ttf",
]

# 3. 영어 전용 폰트 (숫자/영문 학습용)
ENGLISH_FONTS = [
    "apache/roboto/Roboto-Regular.ttf",
    "apache/roboto/Roboto-Bold.ttf",
    "apache/roboto/Roboto-Italic.ttf",
    "ofl/oswald/Oswald-VariableFont_wght.ttf",
    "ofl/lato/Lato-Regular.ttf",
    "ofl/lato/Lato-Bold.ttf",
    "ofl/montserrat/Montserrat-VariableFont_wght.ttf",
    "ofl/opensans/OpenSans-VariableFont_wdth,wght.ttf",
]

# 구글 폰트 기본 경로
BASE_URL = "https://github.com/google/fonts/raw/main/ofl/"
BASE_URL_APACHE = "https://github.com/google/fonts/raw/main/" # 로보토 등은 경로가 다름

def download_file(font_info):
    """개별 파일 다운로드 함수"""
    # URL 조립
    if font_info.startswith("apache"):
        url = f"{BASE_URL_APACHE}{font_info}"
        filename = font_info.split("/")[-1]
    else:
        url = f"{BASE_URL}{font_info}"
        filename = font_info.split("/")[-1]
        
    save_path = os.path.join(SAVE_DIR, filename)

    if os.path.exists(save_path):
        return f"⏭️  [스킵] {filename}"

    try:
        urllib.request.urlretrieve(url, save_path)
        
        # 파일 검증
        if os.path.getsize(save_path) < 1000:
            os.remove(save_path)
            return f"❌ [실패-HTML] {filename}"
            
        return f"✅ [성공] {filename}"
    except Exception as e:
        return f"❌ [에러] {filename}: {e}"

def main():
    # SSL 인증 무시
    ssl._create_default_https_context = ssl._create_unverified_context

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    print(f"🚀 대규모 폰트 다운로드 시작! (총 {len(KOREAN_FONTS) + len(ENGLISH_FONTS)}개 예정)")
    
    # 전체 리스트 합치기
    all_fonts = KOREAN_FONTS + ENGLISH_FONTS
    
    # 멀티스레드로 빠르게 다운로드 (동시 5개)
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(download_file, all_fonts)
        
        for res in results:
            print(res)
            
    print("\n🎉 모든 다운로드 작업 완료!")
    print("👉 generate_data.py를 실행하면 이 폰트들을 사용해 엄청나게 다양한 데이터를 만듭니다.")

if __name__ == "__main__":
    main()