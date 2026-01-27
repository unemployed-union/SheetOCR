from ultralytics import YOLO
import os


def train():
    model = YOLO('./yolov8n.pt')

   # 2. 경로 설정 (프로젝트 폴더 기준)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    print(f"🚀 학습 시작! 저장 경로: {os.path.join(BASE_DIR, 'runs', 'detect')}")

    # 3. 학습 시작 (한 번만 실행!)
    results = model.train(
        data='./dataset_yolo/data.yaml',
        epochs=100,
        imgsz=640,
        device='cuda',

        # [핵심] 결과 저장 경로 고정
        project=os.path.join(BASE_DIR, 'runs', 'detect'),
        name='train_result', 
        exist_ok=True,
        
        # [추가 팁] 데이터가 적을 때 켜면 좋은 옵션들 (선택사항)
        # mosaic=1.0, 
        # degrees=5.0,
    )

    print("✅ 학습이 완료되었습니다.")

if __name__ == "__main__":
    train()
