import json

# 1. metadata.jsonl 파일 경로
metadata_path = "./dataset/train/metadata.jsonl" # 경로 확인 필요
vocab_save_path = "vocab_train.txt"

unique_chars = set()

# 2. 데이터셋의 정답(Label)을 모두 읽어서 글자 수집
with open(metadata_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        # text 키에 정답이 있다고 가정
        text = data.get('text', '') # 또는 'label' 등 키 이름 확인
        for char in text:
            unique_chars.add(char)

# 3. 정렬 (특수문자 -> 한글 순)
sorted_chars = sorted(list(unique_chars))

# 4. 파일로 저장
with open(vocab_save_path, 'w', encoding='utf-8') as f:
    for char in sorted_chars:
        f.write(char + '\n')

print(f"새로운 Vocab 생성 완료! 총 글자 수: {len(sorted_chars)}개")
# 아마 100~500개 사이가 나올 겁니다.