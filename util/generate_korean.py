import string

def make_korean_optimal_vocab():
    vocab = []

    # 1. 영어, 숫자, 특수문자, 공백 (기본)
    vocab += list(string.digits)
    vocab += list(string.ascii_letters)
    vocab += list(string.punctuation)
    vocab.append(" ")

    # 2. 완성형 한글 (KS X 1001) 2,350자
    # 파이썬에서는 'cp949' 인코딩을 이용해서 쉽게 구할 수 있습니다.
    # (믜, 쌰 같은 희귀한 글자는 빠지고, 가, 각, 간... 같은 상용 글자만 들어감)
    korean_chars = []
    for c in range(0xAC00, 0xD7A4): # 유니코드 한글 전체 범위
        char = chr(c)
        try:
            # KS X 1001(euc-kr)에 있는 글자인지 확인
            char.encode('euc-kr') 
            korean_chars.append(char)
        except UnicodeEncodeError:
            pass # 안 쓰이는 희귀 한글은 버림

    print(f"추출된 한글 개수: {len(korean_chars)}자 (예상: 2350자)")
    vocab += korean_chars

    # 3. 중복 제거 및 정렬
    vocab = sorted(list(set(vocab)))
    
    # 4. 저장
    with open("vocab.txt", "w", encoding="utf-8") as f:
        for char in vocab:
            f.write(char + "\n")

    print("-" * 30)
    print(f"✅ 최종 Vocab 생성 완료! 총 {len(vocab)}개")
    print("경로: ./vocab.txt")
    print("-" * 30)

if __name__ == "__main__":
    make_korean_optimal_vocab()