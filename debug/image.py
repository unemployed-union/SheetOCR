import cv2
import numpy as np
import os

def debug_sheet_music_preprocessing(image_path, output_dir="debug_output"):
    # 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = os.path.basename(image_path)
    print(f"Processing: {filename}")

    # 1. 이미지 로드 및 그레이스케일 변환
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{output_dir}/01_gray.jpg", gray)

    # 2. 이진화 (OTSU) - 글씨를 흰색(255)으로 반전
    # thresh_val은 자동 계산된 임계값
    thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(f"{output_dir}/02_binary.jpg", binary)

    # 3. 수평선(오선) 감지 및 제거
    # 이미지 너비의 60% 정도 길이의 커널을 사용 (너무 짧으면 글자도 선으로 인식함)
    img_w = binary.shape[1]
    line_min_width = int(img_w * 0.6) 
    
    # (kernel_width, 1) 크기의 구조 요소 생성
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_min_width, 1))
    
    # Morph Open: 깎았다가 다시 불림 -> 작은 건 사라지고 긴 선만 남음
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    cv2.imwrite(f"{output_dir}/03_lines_only.jpg", detected_lines)

    # 원본에서 선 빼기 (글자만 남기기)
    text_only = cv2.subtract(binary, detected_lines)
    cv2.imwrite(f"{output_dir}/04_text_only.jpg", text_only)

    # 4. 글자 뭉치기 (Dilation)
    # 가로로 길고 세로로 짧은 커널을 써서 옆 글자와 붙게 만듦
    # 파라미터 튜닝 포인트: (20, 5) 숫자를 조절하세요.
    # 가로(20)를 키우면 단어 간격이 멀어도 붙습니다.
    # 세로(5)를 키우면 위아래 줄이 붙어버릴 수 있으니 주의하세요.
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    dilated = cv2.dilate(text_only, dilate_kernel, iterations=3) 
    cv2.imwrite(f"{output_dir}/05_dilated.jpg", dilated)

    # 5. 윤곽선 찾기 및 필터링
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 원본 이미지에 박스 그려서 확인하기 (컬러 변환)
    debug_img = img.copy()
    
    candidates = []
    img_h, img_w = gray.shape

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # --- [튜닝 포인트] 필터링 조건 ---
        # 1. 위치: 상단 30% 이내인가?
        is_top = y < (img_h * 0.3)
        # 2. 크기: 너무 작지 않은가? (가로 10% 이상, 높이 15픽셀 이상)
        is_big = w > (img_w * 0.1) and h > 15
        # 3. 비율: 가로가 세로보다 긴가? (선택사항)
        is_wide = w > h 

        # 조건에 맞으면 초록색 박스, 아니면 빨간색 박스
        if is_top and is_big:
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2) # Green (Candidate)
            candidates.append((x, y, w, h, w*h)) # 면적까지 저장
        else:
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 1) # Red (Ignored)

    cv2.imwrite(f"{output_dir}/06_contours.jpg", debug_img)

    # 6. 최종 제목 선정 (후보 중 면적이 가장 큰 것)
    if candidates:
        # 면적(4번째 인덱스) 기준으로 내림차순 정렬
        candidates.sort(key=lambda x: x[4], reverse=True)
        bx, by, bw, bh, _ = candidates[0]
        
        # 여백 좀 주고 자르기
        padding = 10
        bx = max(0, bx - padding)
        by = max(0, by - padding)
        bw = min(img_w - bx, bw + padding*2)
        bh = min(img_h - by, bh + padding*2)
        
        final_crop = img[by:by+bh, bx:bx+bw]
        cv2.imwrite(f"{output_dir}/07_final_result.jpg", final_crop)
        print("✅ Success: Title found.")
    else:
        print("❌ Fail: No title candidate found.")

# --- 실행 ---
# 테스트할 이미지 경로를 넣으세요
debug_sheet_music_preprocessing("dataset/test_images/sample.png")