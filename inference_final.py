import torch
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms

# 우리가 만든 ViT 모듈들
from vis_transformer.tokenizer import Tokenizer
from vis_transformer.vis_transformer import SimpleViTForOCR

# -------------------------------------------------
# 1. 모델 로딩 (YOLO + ViT)
# -------------------------------------------------
def load_models(yolo_path, vit_path, vocab_file, device):
    # (A) YOLO 로드 (Detector)
    print("👀 Loading Detector (YOLO)...")
    detector = YOLO(yolo_path)

    # (B) ViT 로드 (Recognizer)
    print("🧠 Loading Recognizer (ViT)...")
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab_list = [line.strip('\n') for line in f.readlines()]
    tokenizer = Tokenizer(vocab_list)
    
    recognizer = SimpleViTForOCR(
        vocab_size=len(vocab_list) + 1,
        img_height=112, img_width=448, embed_dim=768
        # 학습 때 쓴 파라미터 확인 필수!
    )
    recognizer.load_state_dict(torch.load(vit_path, map_location=device))
    recognizer.to(device)
    recognizer.eval()

    return detector, recognizer, tokenizer

# -------------------------------------------------
# 2. 파이프라인 실행
# -------------------------------------------------
def get_title_from_sheet(image_path, detector, recognizer, tokenizer, device):
    # (A) 제목 위치 찾기 (Detection)
    # conf=0.25: 확신이 25% 이상인 것만 찾기
    results = detector.predict(image_path, conf=0.25) 
    
    # 찾은 게 없으면 종료
    if len(results[0].boxes) == 0:
        return "❌ 제목을 못 찾았습니다."

    # 여러 개 찾았으면, 가장 위에 있는 놈(y좌표 최소)이 제목일 확률 99%
    # box format: xyxy (x1, y1, x2, y2)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    # y1(세로 위치) 기준으로 정렬해서 맨 위 박스 선택
    boxes = sorted(boxes, key=lambda x: x[1]) 
    best_box = boxes[0] 
    
    x1, y1, x2, y2 = map(int, best_box)
    
    # (B) 이미지 자르기 (Crop)
    original_img = Image.open(image_path).convert('RGB')
    
    # 여백(Padding)을 좀 줘야 글자가 안 잘리고 ViT가 잘 읽음
    padding = 10
    w, h = original_img.size
    crop_box = (
        max(0, x1 - padding), 
        max(0, y1 - padding), 
        min(w, x2 + padding), 
        min(h, y2 + padding)
    )
    title_img = original_img.crop(crop_box)

    # (C) 글자 읽기 (Recognition)
    # ViT용 전처리 (Resize & Normalize)
    transform = transforms.Compose([
        transforms.Resize((112, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    input_tensor = transform(title_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = recognizer(input_tensor)
        pred_indices = output.argmax(dim=2)
        decoded_text = tokenizer.decode(pred_indices[0].tolist())

    return decoded_text, title_img

# -------------------------------------------------
# 실행
# -------------------------------------------------
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 파일 경로들
    YOLO_PATH = "runs/detect/train/weights/best.pt" # YOLO 학습 결과
    VIT_PATH = "best_model.pth"                     # ViT 학습 결과
    VOCAB_PATH = "vocab.txt"
    TEST_IMG = "test_images/은혜.jpg"

    # 로드
    detector, recognizer, tokenizer = load_models(YOLO_PATH, VIT_PATH, VOCAB_PATH, DEVICE)
    
    # 실행
    title_text, cropped_img = get_title_from_sheet(TEST_IMG, detector, recognizer, tokenizer, DEVICE)
    
    print(f"🎵 추출된 제목: {title_text}")
    
    # 잘린 이미지 확인
    cropped_img.show()