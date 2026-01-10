import os
import requests
import zipfile
import io
import shutil

# --- 설정 ---
DOWNLOAD_DIR = "fonts/ko"  # 폰트 저장할 폴더

# 구글 폰트 다운로드 링크 모음 (엄선된 고퀄리티 한글 폰트)
# 이름: (가족명, 다운로드 URL 파라미터)
FONT_URLS = {
    "NotoSansKR": "Noto+Sans+KR",
    "NotoSerifKR": "Noto+Serif+KR",
    "NanumGothic": "Nanum+Gothic",
    "NanumMyeongjo": "Nanum+Myeongjo",
    "NanumPenScript": "Nanum+Pen+Script",
    "NanumBrushScript": "Nanum+Brush+Script",
    "GowunDodum": "Gowun+Dodum",
    "GowunBatang": "Gowun+Batang",
    "DoHyeon": "Do+Hyeon",
    "Jua": "Jua",
    "YeonSung": "Yeon+Sung",
    "Sunflower": "Sunflower",
    "GothicA1": "Gothic+A1",
    "HiMelody": "Hi+Melody",
    "GamjaFlower": "Gamja+Flower",
    "BlackHanSans": "Black+Han+Sans",
    "SongMyung": "Song+Myung",
    "CuteFont": "Cute+Font",
    "Gaegu": "Gaegu",
    "Dokdo": "Dokdo",
    "EastSeaDokdo": "East+Sea+Dokdo",
}

BASE_URL = "https://fonts.google.com/download?family="

def download_and_extract_fonts():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    
    print(f"🚀 한글 폰트 다운로드 시작...")
    
    # [핵심] 봇 차단 회피용 헤더 (크롬 브라우저인 척 위장)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    success_count = 0

    for name, param in FONT_URLS.items():
        url = BASE_URL + param
        print(f"⬇️ Downloading: {name}...")
        
        try:
            # headers=headers 추가
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # [디버깅] 만약 또 에러나면, 서버가 뭘 줬는지 확인하는 코드
            # ZIP 파일 시그니처(PK..)로 시작하지 않으면 에러 처리
            if not response.content.startswith(b'PK'):
                print(f"  ⚠️ 실패: 서버가 ZIP 대신 다른 걸 줬습니다. (내용: {response.content[:20]}...)")
                continue

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                for filename in z.namelist():
                    if filename.lower().endswith(('.ttf', '.otf')):
                        target_filename = os.path.basename(filename)
                        if not target_filename: continue
                        
                        target_path = os.path.join(DOWNLOAD_DIR, f"{name}_{target_filename}")
                        with open(target_path, 'wb') as f:
                            f.write(z.read(filename))
                            
            print(f"  ✅ 성공")
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 에러 발생: {e}")

    print("-" * 30)
    print(f"🎉 총 {success_count}개 폰트 다운로드 완료!")

if __name__ == "__main__":
    download_and_extract_fonts()