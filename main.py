import torch
import pandas as pd
from transformers import BertTokenizer
from torch import nn
from train import BertClassifier  # 假定BertClassifier类在main.py中
import numpy as np

# 标签映射，与训练时一致
labels = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4, '时尚': 5, '时政': 6, '游戏': 7, '科技': 8, '财经': 9}
id2label = {v: k for k, v in labels.items()}

tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')


def predict(csv_path, model_path='bert_classifier.pth', out_path='predict_result.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化模型结构，并加载参数
    model = BertClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 加载csv
    # df = pd.read_csv(csv_path)
    df = pd.read_csv(csv_path, header=None, sep='\t')
    df.columns = ['text']
    print(df.head())

    all_texts = df['text'].tolist()
    predictions = []

    with torch.no_grad():
        for text in all_texts:
            encoding = tokenizer(
                text,
                padding='max_length',
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            output = model(input_ids, attention_mask)
            pred_label_id = torch.argmax(output, dim=1).item()
            # 映射回标签名
            pred_label = id2label[pred_label_id]
            predictions.append(pred_label)
    # 添加预测字段并保存
    dfresult = pd.DataFrame(columns=['predict_category'])
    #    text category predict
    # (0 rows)
    dfresult['predict_category'] = predictions
    dfresult.to_csv(out_path, index=False, header=False)
    print(f"预测完成，已保存到 {out_path}")


if __name__ == '__main__':
    # 用法举例:
    # python main.py --csv your_input.csv
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='需要预测的csv文件路径')
    parser.add_argument('--output', type=str, default='predict_result.csv', help='输出路径')
    parser.add_argument('--task', type=str, default='2', help='问题序号')
    args = parser.parse_args()
    predict(args. input, './bert_classifier.pth', args.output)