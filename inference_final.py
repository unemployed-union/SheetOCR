import torch
from ultralytics import YOLO
from PIL import Image, ImageOps
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
        vocab_list = [line.replace('\n', '').replace('\r', '') for line in f.readlines()]
    tokenizer = Tokenizer(vocab_list)
    
    recognizer = SimpleViTForOCR(
        vocab_size=len(vocab_list) + 1,
        img_height=112, img_width=448, embed_dim=384
        # 학습 때 쓴 파라미터 확인 필수!
    )
    recognizer.load_state_dict(torch.load(vit_path, map_location=device))
    recognizer.to(device)
    recognizer.eval()

    return detector, recognizer, tokenizer

# -------------------------------------------------
# 2. 파이프라인 실행
# -------------------------------------------------
# def get_title_from_sheet(image_path, detector, recognizer, tokenizer, device):
#     # (A) 제목 위치 찾기 (Detection)
#     # conf=0.25: 확신이 25% 이상인 것만 찾기
#     results = detector.predict(image_path, conf=0.25) 
    
#     # 찾은 게 없으면 종료
#     if len(results[0].boxes) == 0:
#         return "❌ 제목을 못 찾았습니다.", None

#     # 여러 개 찾았으면, 가장 위에 있는 놈(y좌표 최소)이 제목일 확률 99%
#     # box format: xyxy (x1, y1, x2, y2)
#     boxes = results[0].boxes.xyxy.cpu().numpy()
    
#     # y1(세로 위치) 기준으로 정렬해서 맨 위 박스 선택
#     boxes = sorted(boxes, key=lambda x: x[1]) 
#     best_box = boxes[0] 
    
#     x1, y1, x2, y2 = map(int, best_box)
    
#     # (B) 이미지 자르기 (Crop)
#     original_img = Image.open(image_path).convert('RGB')
    
#     # 여백(Padding)을 좀 줘야 글자가 안 잘리고 ViT가 잘 읽음
#     padding = 10
#     w, h = original_img.size
#     crop_box = (
#         max(0, x1 - padding), 
#         max(0, y1 - padding), 
#         min(w, x2 + padding), 
#         min(h, y2 + padding)
#     )
#     title_img = original_img.crop(crop_box)

#     # (C) 글자 읽기 (Recognition)
#     # ViT용 전처리 (Resize & Normalize)
#     transform = transforms.Compose([
#         transforms.Resize((112, 448)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])
    
#     input_tensor = transform(title_img).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         output = recognizer(input_tensor)
#         pred_indices = output.argmax(dim=2)
#         decoded_text = tokenizer.decode(pred_indices[0].tolist())

#     return decoded_text, title_img

def get_title_from_sheet(image_path, yolo_model, vit_model, tokenizer, device):
    # 1. YOLO로 제목 영역 찾기
    results = yolo_model(image_path)
    
    # YOLO가 못 찾았을 경우 예외 처리
    if not results or len(results[0].boxes) == 0:
        print("⚠️ YOLO가 제목을 찾지 못했습니다.")
        return "Unknown"

    # 가장 확신하는 박스 하나 가져오기
    box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # 2. 이미지 로드 및 자르기 (YOLO 좌표대로 Crop)
    original_img = Image.open(image_path).convert("RGB")
    cropped_img = original_img.crop((x1, y1, x2, y2))
    
    # ---------------------------------------------------------
    # [핵심 수정] 448x112로 "비율 유지하면서" 맞추기 (Smart Resize)
    # ---------------------------------------------------------
    target_h = 112
    target_w = 448
    
    # A. 현재 이미지 비율 계산
    w, h = cropped_img.size
    ratio = target_h / h
    new_w = int(w * ratio)
    
    # B. 높이를 112로 맞추고, 너비는 비율대로 리사이즈
    resized_img = cropped_img.resize((new_w, target_h), Image.Resampling.BILINEAR)
    
    # C. 빈 캔버스(흰색 배경) 만들기 (ViT 입력 크기인 448x112)
    final_img = Image.new("L", (target_w, target_h), 255) # 255=흰색
    
    # D. 리사이즈된 이미지를 캔버스에 붙이기
    if new_w > target_w:
        # 만약 글자가 너무 길면 -> 448로 강제 축소 (어쩔 수 없음)
        resized_img = resized_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        final_img.paste(resized_img.convert('L'), (0, 0))
    else:
        # 글자가 짧으면 -> 왼쪽(0,0)에 붙이고 나머지는 흰 여백으로 둠
        final_img.paste(resized_img.convert('L'), (0, 0))
        
    # ---------------------------------------------------------
    
    # 3. 텐서 변환 및 정규화 (학습 때와 동일하게!)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # [중요] 정규화 필수
    ])
    
    input_tensor = transform(final_img).unsqueeze(0).to(device)
    
    # 4. ViT 모델 추론
    vit_model.eval()
    with torch.no_grad():
        output = vit_model(input_tensor) # (Batch, Seq, Vocab)
        probabilities = torch.softmax(output, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        
    # 5. 결과 디코딩
    predicted_text = tokenizer.decode(predictions[0].tolist())
    return predicted_text, final_img

# -------------------------------------------------
# 실행
# -------------------------------------------------
if __name__ == "__main__":
    # 1. 설정
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 파일 경로들 (본인 경로에 맞게 확인!)
    YOLO_PATH = "runs/detect/train_result/weights/best.pt" # 혹은 train_result_hardcore 등
    VIT_PATH = "best_model_one.pth"
    VOCAB_PATH = "vocab.txt"
    TEST_IMG = "test_images/은혜.jpg" # 테스트하고 싶은 이미지

    # 2. 모델 로드
    detector, recognizer, tokenizer = load_models(YOLO_PATH, VIT_PATH, VOCAB_PATH, DEVICE)
    
    # 3. 실행 (제목 추출)
    print(f"🖼️ 이미지 분석 시작: {TEST_IMG}")
    title_text, cropped_img = get_title_from_sheet(TEST_IMG, detector, recognizer, tokenizer, DEVICE)
    
    print("-" * 30)
    print(f"🎵 [결과] 추출된 텍스트: {title_text}")
    print("-" * 30)
    
    # 4. [핵심] ViT가 본 이미지를 파일로 저장 (디버깅용)
    if cropped_img is not None:
        save_path = "debug_crop_result.jpg"
        cropped_img.save(save_path)
        print(f"📸 잘린 이미지를 저장했습니다 -> {save_path}")
        print("👉 VS Code 왼쪽 파일 탐색기에서 'debug_crop_result.jpg'를 클릭해서 확인해보세요.")
        
        # 추가 조언
        print("\n[진단 가이드]")
        print("1. 이미지가 제목이 아니라 '가사'나 '음표'인가요? -> YOLO가 범인입니다. (YOLO 재학습 필요)")
        print("2. 이미지는 제목이 맞는데 텍스트가 이상한가요? -> ViT가 범인입니다. (ViT 데이터 추가 학습 필요)")
    else:
        print("⚠️ YOLO가 아무것도 찾지 못해서 저장할 이미지가 없습니다.")