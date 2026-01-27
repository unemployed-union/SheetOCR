import string

def make_korean_optimal_vocab():
    print("🚀 Vocab 생성 시작 (Byte Decoding Mode)...")
    vocab = []

    # 1. 영어, 숫자, 특수문자, 공백
    vocab += list(string.digits)
    vocab += list(string.ascii_letters)
    vocab += list(string.punctuation)
    vocab.append(" ")
    
    print(f"   ▶ 영어/숫자/특수문자: {len(vocab)}개")

    # 2. 완성형 한글 (KS X 1001) 2,350자 강제 복원
    # 원리: KS X 1001 표준에서 한글은 아래 바이트 범위에 정확히 매핑되어 있습니다.
    # - 첫 번째 바이트: 0xB0 ~ 0xC8 (행)
    # - 두 번째 바이트: 0xA1 ~ 0xFE (열)
    # 이 범위를 순회하며 decoding하면 무조건 2,350자가 나옵니다.
    
    korean_chars = []
    
    # 0xB0(176) ~ 0xC8(200)
    for h in range(0xB0, 0xC9): 
        # 0xA1(161) ~ 0xFE(254)
        for l in range(0xA1, 0xFF): 
            try:
                # 바이트를 직접 조립해서 글자로 변환
                char = bytes([h, l]).decode('euc-kr')
                korean_chars.append(char)
            except:
                pass

    print(f"   ▶ 완성형 한글 추출 완료: {len(korean_chars)}개 (목표: 2350)")
    
    if len(korean_chars) != 2350:
        print("   ❌ 에러: 여전히 2350자가 아닙니다. 개발 환경을 점검해야 합니다.")
    
    vocab += korean_chars

    # 3. 중복 제거 및 정렬
    vocab = sorted(list(set(vocab)))

    # 4. 저장
    with open("vocab.txt", "w", encoding="utf-8") as f:
        for char in vocab:
            if char == "\n": continue
            f.write(char + "\n")

    print("-" * 30)
    print(f"🎉 최종 Vocab 생성 완료!")
    print(f"📊 총 글자 수: {len(vocab)} (한글 2350 + 영어/특수문자 = 2440~2450개 예상)")
    print("-" * 30)

if __name__ == "__main__":
    make_korean_optimal_vocab()