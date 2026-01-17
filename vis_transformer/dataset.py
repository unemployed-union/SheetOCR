import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

class SheetMusicDataset(Dataset):
    def __init__(self, root_dir, df, tokenizer, transform=None):
        self.root_dir = root_dir
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform # 정규화(Normalize)만 남김
        
        self.cached_tensors = [] # 이미지가 아니라 '텐서'를 저장
        self.cached_targets = [] # 정답도 미리 텐서로 변환해서 저장
        
        print(f"🔄 데이터 {len(df)}장 텐서 변환 및 RAM 캐싱 중... (최적화)")
        
        resize_tool = Image.BICUBIC
        target_size = (448, 112)
        
        for idx in tqdm(range(len(df))):
            try:
                # 1. 이미지 로드 & 리사이즈
                file_name = df.iloc[idx]['file_name']
                img_path = os.path.join(self.root_dir, file_name)
                img = Image.open(img_path).convert("L")
                img = img.resize(target_size, resample=resize_tool)
                
                # 2. [핵심] 바로 텐서(UInt8)로 변환해 저장!
                # transforms.ToTensor()를 안 쓰고 numpy로 바꾼 뒤 torch로 감쌉니다.
                # (H, W, C) -> (C, H, W) 순서 변경
                img_np = np.array(img)
                img_tensor = torch.from_numpy(img_np).unsqueeze(0) # dtype=torch.uint8 (가벼움)

                self.cached_tensors.append(img_tensor)
                
                # 3. 정답(Target)도 미리 텐서로 변환
                text = df.iloc[idx]['text']
                if hasattr(self.tokenizer, 'encode'):
                    encoded = self.tokenizer.encode(text)
                else:
                    encoded = [self.tokenizer.token_to_id.get(c, 0) for c in text]
                self.cached_targets.append(torch.tensor(encoded, dtype=torch.long))
                
            except Exception as e:
                print(f"Error: {e}")
                # 에러 시 검은색 텐서 추가
                dummy = torch.zeros((3, 112, 448), dtype=torch.uint8)
                self.cached_tensors.append(dummy)
                self.cached_targets.append(torch.tensor([], dtype=torch.long))
                
        print("✅ 캐싱 완료! (CPU 부하 최소화)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. 캐싱된 텐서 꺼내기 (아주 빠름)
        # uint8 (0~255) 상태
        image_tensor = self.cached_tensors[idx]
        target = self.cached_targets[idx]
        
        # 2. Float 변환 (0~1) : 나누기 연산만 하면 됨
        # div(255)는 매우 빠름
        image = image_tensor.float().div(255.0)
        
        # 3. Normalize 적용 (transform에 Normalize만 있어야 함)
        if self.transform:
            image = self.transform(image)

        return image, target

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = torch.cat(targets, dim=0)
    target_lengths = torch.tensor([len(t) for t in list(zip(*batch))[1]], dtype=torch.long)
    return images, targets, target_lengths