import re

def calculate(lyrics: list):
    """Calculate the percentage of English words in the lyrics."""
    if not lyrics:
        return 0.0

    all_words = " ".join(lyrics).split()
    total_words = len(all_words)
    if total_words == 0:
        return 0.0

    english_words = [word for word in all_words if re.match(r'^[a-zA-Z]+$', word)]
    english_word_count = len(english_words)

    return (english_word_count / total_words) * 100

if __name__ == "__main__":
    # 從使用者輸入獲取歌詞``
    print("請輸入歌詞，每行以 Enter 分隔，輸入 '#' 結束：")
    lyrics = []
    while True:
        line = input()
        if line.strip() == "#":
            break
        lyrics.append(line)

    # 計算英文單字比例
    percentage = calculate(lyrics)
    print(f"英文單字佔比: {percentage:.2f}%")