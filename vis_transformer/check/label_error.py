# check_label_error.py
import sys
# main.py가 있는 폴더에서 실행한다고 가정
from ..tokenizer import Tokenizer 

def test_tokenizer():
    print("🕵️‍♂️ 토크나이저 정밀 검사를 시작합니다...")
    
    # 1. Vocab 로드
    try:
        with open("vocab.txt", "r", encoding="utf-8") as f:
            vocab_list = [line.strip('\n') for line in f.readlines()]
        print(f"✅ vocab.txt 로드 성공 (총 {len(vocab_list)}개 토큰)")
    except FileNotFoundError:
        print("❌ vocab.txt 파일이 없습니다!")
        return

    # 2. 토크나이저 생성
    tokenizer = Tokenizer(vocab_list)
    
    # 3. 문제의 텍스트 변환 테스트
    target_text = "434곡"
    print(f"\n🧪 테스트 문구: '{target_text}'")
    
    # dataset.py와 똑같은 로직으로 인코딩
    if hasattr(tokenizer, 'encode'):
        encoded_ids = tokenizer.encode(target_text)
        print("👉 encode() 함수를 사용했습니다.")
    else:
        print("❌ Tokenizer에 encode 함수가 없습니다! 코드 확인이 필요합니다.")
        return

    print(f"🔢 변환된 ID 리스트: {encoded_ids}")
    
    # 4. 결과 분석 (핵심!)
    print("-" * 30)
    for char, token_id in zip(target_text, encoded_ids):
        # 0번은 보통 [PAD] 이거나 [Blank]
        interpretation = "✅ 정상"
        if token_id == 0:
            interpretation = "🚨 범인! (Blank/PAD로 변환됨)"
        elif token_id == 1 and vocab_list[1] == " ": # 혹시 1번이 공백이면
            interpretation = "⚠️ 주의 (공백으로 변환됨)"
            
        print(f"글자 '{char}' \t-> ID: {token_id} \t {interpretation}")
    print("-" * 30)

    # 5. 최종 진단
    if any(id == 0 for id in encoded_ids if id != encoded_ids[-1]):
        print("\n[📢 진단 결과]")
        print("범인은 '토크나이저'입니다! 😡")
        print("숫자 '4'가 ID 0(Blank)으로 바뀌어버려서, 모델이 숫자를 '투명 인간' 취급하고 있습니다.")
        print("해결책: tokenizer.py 파일의 encode 함수 로직을 고쳐야 합니다.")
    else:
        print("\n[📢 진단 결과]")
        print("토크나이저는 정상입니다. 그렇다면... 정말 미스터리네요.")
        print("이 결과가 나오면 tokenizer.py 파일 내용을 보여주세요.")

if __name__ == "__main__":
    test_tokenizer()