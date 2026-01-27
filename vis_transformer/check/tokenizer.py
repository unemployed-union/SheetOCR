import sys
from .tokenizer import Tokenizer 

# 1. Vocab 로드
vocab_path = "vocab.txt"
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab_list = [line.strip('\n') for line in f.readlines()]

tokenizer = Tokenizer(vocab_list)

# 2. 숫자 테스트 (434곡)
text = "434곡"
print(f"입력 텍스트: '{text}'")
print("-" * 30)

# [수정] 직접 변수(token_to_id)를 찾지 말고, encode 함수를 씁니다.
try:
    if hasattr(tokenizer, 'encode'):
        encoded = tokenizer.encode(text)
    else:
        # encode 함수가 없다면, 내부 변수를 뒤져서 찾아냅니다.
        # (stoi, char2idx, token_to_id 등 흔한 이름 탐색)
        mapping = getattr(tokenizer, 'stoi', 
                  getattr(tokenizer, 'char2idx', 
                  getattr(tokenizer, 'token_to_id', None)))
        
        if mapping:
            encoded = [mapping.get(c, 0) for c in text]
        else:
            print("❌ Tokenizer 내부 변수 이름을 못 찾겠습니다.")
            print("가진 속성들:", dir(tokenizer))
            sys.exit(1)

    # 3. 결과 분석
    for char, token_id in zip(text, encoded):
        status = "✅ 정상" if token_id != 0 else "🚨 범인! ([PAD]로 변환됨)"
        print(f"글자 '{char}' -> ID: {token_id} \t {status}")

    print("-" * 30)
    print(f"최종 라벨: {encoded}")

    # 4. 진단
    # (주의: encoded 리스트 길이와 text 길이가 다를 수 있으므로 체크)
    numeric_ids = [eid for char, eid in zip(text, encoded) if char.isdigit()]
    
    if all(nid == 0 for nid in numeric_ids) and len(numeric_ids) > 0:
        print("\n[진단 결과]")
        print("👉 Tokenizer가 숫자를 전부 0([PAD])으로 바꾸고 있습니다!")
        print("👉 vocab.txt 로딩이 잘못되었거나, Tokenizer가 숫자를 무시하도록 설정된 것 같습니다.")
    elif 0 in numeric_ids:
         print("\n[진단 결과]")
         print("👉 일부 숫자가 0으로 변환됩니다. Vocab을 확인하세요.")
    else:
        print("\n[진단 결과]")
        print("✅ Tokenizer는 숫자를 정상적으로 인식하고 있습니다.")
        print("👉 그렇다면 '이미지 전처리(투명도)' 문제가 맞을 확률이 높습니다.")

except Exception as e:
    print(f"테스트 중 에러 발생: {e}")
    # 혹시 모르니 Tokenizer 속성을 다 보여줍니다.
    print("🔍 Tokenizer가 가진 속성 목록:", tokenizer.__dict__.keys())