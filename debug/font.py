import os
import glob
from fontTools.ttLib import TTFont

# 여기에 폰트 폴더 경로를 입력하세요 (예: "fonts")
FONT_DIR = "fonts"

def check_fonts_detail(font_dir):
    print(f"🔍 '{font_dir}' 폴더 정밀 진단 시작...")
    
    # 1. 파일 찾기
    extensions = ['**/*.ttf', '**/*.otf', '**/*.ttc']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(font_dir, ext), recursive=True))
    
    files = sorted(list(set(files)))
    
    if not files:
        print("❌ 폰트 파일을 하나도 못 찾았습니다. 경로를 다시 확인하세요.")
        return

    print(f"📂 총 {len(files)}개의 폰트 파일 발견. 검사 시작!\n")
    
    success = 0
    fail = 0

    for idx, fpath in enumerate(files):
        filename = os.path.basename(fpath)
        try:
            # 2. 폰트 열기 시도
            if fpath.lower().endswith('.ttc'):
                font = TTFont(fpath, fontNumber=0) # TTC는 첫 번째 폰트 로드
            else:
                font = TTFont(fpath)
            
            # 3. 문자표(Cmap) 추출 시도
            cmap = font.getBestCmap()
            
            if cmap is None:
                print(f"[{idx+1}] ⚠️ {filename}: 로드는 됐는데 문자표(CMAP)가 없습니다. (사용 불가)")
                fail += 1
            elif len(cmap) == 0:
                print(f"[{idx+1}] ⚠️ {filename}: 문자표가 비어 있습니다. (글자가 하나도 없음)")
                fail += 1
            else:
                print(f"[{idx+1}] ✅ {filename}: 정상 (지원 문자 {len(cmap)}개)")
                success += 1
                
        except Exception as e:
            # 4. 에러 발생 시 구체적인 이유 출력
            print(f"[{idx+1}] ❌ {filename}: 에러 발생 -> {e}")
            fail += 1

    print("\n" + "="*50)
    print(f"📊 최종 결과: 성공 {success}개 / 실패 {fail}개")
    print("="*50)

if __name__ == "__main__":
    check_fonts_detail(FONT_DIR)