import torch
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# [핵심] WSL 등 화면 없는 환경을 위한 설정 (창 띄우기 금지)
import matplotlib
matplotlib.use('Agg') 

# 1. 파일 경로 설정
MODEL_PATH = "runs/detect/train_result/weights/best.pt"
TEST_IMG = "test_images/은혜.jpg"
SAVE_PATH = "debug_result.png"  # 결과를 여기로 저장합니다.

def debug_yolo():
    print(f"👀 모델 로딩 중: {MODEL_PATH}")
    
    # 모델 파일 존재 확인
    if not os.path.exists(MODEL_PATH):
        print(f"❌ [에러] 모델 파일이 없습니다: {MODEL_PATH}")
        return

    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    print(f"🖼️ 이미지 분석 중: {TEST_IMG}")
    
    # [핵심] conf를 0.05(5%)까지 아주 낮춰서 확인
    # save=True를 하면 runs/detect/predict 폴더에 자동 저장되지만, 
    # 확실한 확인을 위해 수동으로 그립니다.
    results = model.predict(TEST_IMG, conf=0.05)
    
    # 결과 분석
    boxes = results[0].boxes
    print("-" * 50)
    print(f"📊 탐지된 객체 수: {len(boxes)}개")
    print("-" * 50)

    if len(boxes) == 0:
        print("💀 결과: 아무것도 못 찾았습니다.")
        print("👉 원인: 학습 데이터 부족으로 과적합(Overfitting) 되었거나, 학습이 덜 됨.")
    else:
        for i, box in enumerate(boxes):
            conf = box.conf.item()
            cls = int(box.cls.item())
            xyxy = box.xyxy[0].tolist()
            print(f"[{i+1}] 클래스: {cls} | 확신도(Conf): {conf:.4f} | 좌표: {xyxy}")

    # 🖼️ 결과 이미지 파일로 저장
    print(f"\n💾 결과 이미지를 저장하는 중... -> {SAVE_PATH}")
    
    # YOLO가 제공하는 plot() 함수로 박스가 그려진 이미지를 가져옴 (numpy array)
    plotted_img = results[0].plot()
    
    # OpenCV로 저장 (BGR 색상이므로 그대로 저장하면 됨)
    cv2.imwrite(SAVE_PATH, plotted_img)
    
    print("✅ 저장 완료! 탐색기에서 'debug_result.png' 파일을 열어보세요.")

if __name__ == "__main__":
    debug_yolo()