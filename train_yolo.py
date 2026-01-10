from ultralytics import YOLO

def train():
    model = YOLO('./yolov8n.pt')

    result = model.train(data='./dataset_yolo/data.yaml', epochs=50, imgsz=640, device='mps')

if __name__ == "__main__":
    train()