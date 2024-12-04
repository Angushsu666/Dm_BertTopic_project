import os
import pandas as pd
from glob import glob

def combine_monthly_data(input_dir, output_file):
    # 加載所有按月數據
    files = glob(f"{input_dir}/*-*.csv")
    monthly_data = pd.concat([pd.read_csv(file) for file in files])

    # 確保 created_utc 是日期格式
    monthly_data['created_utc'] = pd.to_datetime(monthly_data['created_utc'])

    # 添加年份列
    monthly_data['year'] = monthly_data['created_utc'].dt.year

    # 合併按 subreddit 和年份組織數據
    yearly_text = (
        monthly_data.groupby(['subreddit', 'year'])
        .apply(lambda x: ' '.join(x['title'].fillna('') + ' ' + x['selftext'].fillna('') + ' ' + x['body'].fillna('')))
        .reset_index(name='text')
    )

    # 檢查目標目錄是否存在，若不存在則創建
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存到文件
    yearly_text.to_csv(output_file, index=False)

if __name__ == "__main__":
    combine_monthly_data(input_dir="data/decompress", output_file="data/yearly/yearly_text.csv")
