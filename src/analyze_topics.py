import os
import re
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd
import json
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def analyze_topics(input_file, output_file):
    # 確保輸出目錄存在
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # 加載年度數據
    yearly_data = pd.read_csv(input_file)

    # 自定義停用詞
    custom_stop_words = set(ENGLISH_STOP_WORDS).union({
        "you", "he", "but", "or", "post", "just", "it", "this", "that",
        "like", "dont", "im", "hes", "people", "removed", "youre",
        "deleted", "thats", "fuck", "game", "good", "year", "team",
        "time", "fucking", "think", "thinks", "got", "make", "doesnt",
        "submission", "subreddit"
    })

    # 文本預處理函數
    def preprocess_text(text):
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # 移除網址
        text = re.sub(r"[^\w\s]", "", text)  # 移除特殊字符
        text = ' '.join([word for word in text.split() if word.lower() not in custom_stop_words])  # 移除停用詞
        return text

    # 合併每個 Subreddit 和年份的文本
    grouped_text = (
        yearly_data.groupby(['subreddit', 'year'])['text']
        .apply(lambda texts: ' '.join(preprocess_text(text) for text in texts))
        .reset_index(name='text')
    )

    # 初始化 UMAP 和 HDBSCAN 模型
    umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.1, metric='cosine')
    hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=2, metric='euclidean')
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, low_memory=True)

    # 保存每個 Subreddit 和年份的熱詞
    hot_words_per_year = {}

    # 為每個 Subreddit 單獨建模
    for subreddit, group in grouped_text.groupby('subreddit'):
        all_texts = group['text'].tolist()
        all_texts = [text for text in all_texts if isinstance(text, str) and len(text.split()) > 5]  # 過濾短文本

        if len(all_texts) < 2:
            print(f"Skipping subreddit={subreddit} due to insufficient text data.")
            continue

        print(f"Starting topic modeling for subreddit={subreddit} with {len(all_texts)} documents...")
        try:
            topics, _ = topic_model.fit_transform(all_texts)
        except ValueError as e:
            print(f"Error while modeling subreddit={subreddit}: {e}")
            continue

        # 提取主題關鍵詞
        topic_words = topic_model.get_topics()
        for i, row in group.iterrows():
            year = row['year']
            topic_id = topics[i]
            if topic_id == -1:
                print(f"No topic assigned for subreddit={subreddit}, year={year}.")
                continue

            # 獲取前10個熱詞和分數
            hot_words_with_scores = [
                (word, score) for word, score in topic_words.get(topic_id, [])
                if word.lower() not in custom_stop_words
            ][:10]

            if hot_words_with_scores:
                hot_words_per_year[f"{subreddit}_{year}"] = hot_words_with_scores
                print(f"Hot words for subreddit={subreddit}, year={year}: {hot_words_with_scores}")
            else:
                print(f"No meaningful hot words for subreddit={subreddit}, year={year}.")

    # 保存到 JSON 文件
    if hot_words_per_year:
        with open(output_file, "w") as f:
            json.dump(hot_words_per_year, f, indent=4)
        print(f"Hot words successfully saved to {output_file}.")
    else:
        print("No hot words found. Please check the input data or topic modeling process.")


if __name__ == "__main__":
    analyze_topics(
        input_file="data/yearly/yearly_text.csv",
        output_file="data/topics/hot_words_per_year.json"
    )
