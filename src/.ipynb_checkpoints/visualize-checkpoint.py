import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

def create_wordcloud(subreddit, year, hot_words, output_dir):
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 生成文字雲
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(hot_words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"{subreddit} - {year} Hot Words")
    plt.axis('off')

    # 保存圖片
    plt.savefig(f"{output_dir}/{subreddit}_{year}_wordcloud.png")
    plt.close()

def create_bar_chart(subreddit, year, hot_words_with_scores, output_dir):
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 解包熱詞和它們的頻率分數
    words, scores = zip(*hot_words_with_scores)

    plt.figure(figsize=(10, 5))
    plt.bar(words, scores)
    plt.title(f"{subreddit} - {year} Hot Words Frequency")
    plt.xticks(rotation=45)
    plt.xlabel("Words")
    plt.ylabel("Scores")

    # 保存圖片
    plt.savefig(f"{output_dir}/{subreddit}_{year}_barchart.png")
    plt.close()


if __name__ == "__main__":
    # 加載熱詞數據
    with open("data/topics/hot_words_per_year.json", "r") as f:
        hot_words_per_year = json.load(f)

    # 遍歷熱詞並生成可視化
    for key, hot_words_with_scores in hot_words_per_year.items():
        # 分離鍵中的 subreddit 和 year
        subreddit, year = key.rsplit("_", 1)
        hot_words = [word for word, _ in hot_words_with_scores]  # 提取詞語
        create_wordcloud(subreddit, year, hot_words, output_dir="plots/wordclouds")
        create_bar_chart(subreddit, year, hot_words_with_scores, output_dir="plots/bar_charts")
