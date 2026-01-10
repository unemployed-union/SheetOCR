class Tokenizer:
    def __init__(self, char_list):
        # char_list: 데이터셋에 등장하는 모든 글자들의 리스트 (예: ['a', 'b', '가', '나', ...])
        # 0번은 CTC Blank를 위해 비워둡니다.
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(char_list)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(char_list)}
        self.blank_token = 0
        
    def encode(self, text):
        """텍스트 -> 숫자 리스트 변환"""
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]
    
    def decode(self, indices):
        """숫자 리스트 -> 텍스트 변환 (CTC 디코딩 로직 포함)"""
        result = []
        # 1. 중복 제거 및 Blank(0) 제거
        prev_idx = -1
        for idx in indices:
            if idx != prev_idx and idx != self.blank_token:
                result.append(self.idx_to_char.get(idx, "")) # 없는 글자면 무시
            prev_idx = idx
        return "".join(result)
        
    def get_vocab_size(self):
        # 글자 수 + Blank 토큰(1개)
        return len(self.char_to_idx) + 1