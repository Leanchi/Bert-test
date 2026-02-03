
from transformers import BertModel
import numpy as np
from transformers import BertTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm


# 指定本地文件夹的路径
local_model_path = "./bert-base-chinese"

tokenizer = BertTokenizer.from_pretrained(local_model_path)
labels = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4, '时尚': 5, '时政': 6, '游戏': 7, '科技': 8, '财经': 9}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        # self.labels = [labels[label] for label in df['category']]
        # self.texts = [tokenizer(text,
        #                         padding='max_length',
        #                         max_length = 512,
        #                         truncation=True,
        #                         return_tensors="pt")
        #               for text in df['text']]

        # 1. 只转换标签（这个很快，毫秒级）
        self.labels = [labels[label] for label in df['category']]
        # 2. 【关键修改】只保存原始文本字符串，不进行分词
        self.texts = df['text'].tolist()

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        # batch_texts = self.get_batch_texts(idx)
        # batch_y = self.get_batch_labels(idx)
        # return batch_texts, batch_y

        # 3. 只有当程序需要这条数据时，才进行分词
        text = self.get_batch_texts(idx)
        label = self.get_batch_labels(idx)

        # 实时分词
        encoding = tokenizer(
            text,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )

        # tokenizer 返回的 tensor 形状是 [1, 512]，我们需要 [512]
        # 所以需要 squeeze(0) 去掉那个 1 维度
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 返回可以在 GPU 上直接用的 tensor   input_ids, attention_mask, torch.tensor(label)
        return encoding, label


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 10)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer




# def train(model, train_data, val_data, learning_rate, epochs):
#     # 通过Dataset类获取训练和验证集
#     train, val = Dataset(train_data), Dataset(val_data)
#     # DataLoader根据batch_size获取数据，训练时选择打乱样本
#     train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True, num_workers=2)
#     val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
#     # 判断是否使用GPU
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#     # device = torch.device("cpu")
#     # 定义损失函数和优化器
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=learning_rate)
#
#     if use_cuda:
#         model = model.cuda()
#         criterion = criterion.cuda()
#     # 开始进入训练循环
#     for epoch_num in range(epochs):
#         # 定义两个变量，用于存储训练集的准确率和损失
#         total_acc_train = 0
#         total_loss_train = 0
#
#         # 进度条函数tqdm
#         for train_input, train_label in tqdm(train_dataloader):
#             train_label = train_label.to(device)
#             mask = train_input['attention_mask'].to(device)
#             input_id = train_input['input_ids'].squeeze(1).to(device)
#             # 通过模型得到输出
#             output = model(input_id, mask)
#             # 计算损失
#             batch_loss = criterion(output, train_label)
#             total_loss_train += batch_loss.item()
#             # 计算精度
#             acc = (output.argmax(dim=1) == train_label).sum().item()
#             total_acc_train += acc
#             # 模型更新
#             model.zero_grad()
#             batch_loss.backward()
#             optimizer.step()
#         # ------ 验证模型 -----------
#         # 定义两个变量，用于存储验证集的准确率和损失
#         total_acc_val = 0
#         total_loss_val = 0
#         # 不需要计算梯度
#         with torch.no_grad():
#             # 循环获取数据集，并用训练好的模型进行验证
#             for val_input, val_label in val_dataloader:
#                 # 如果有GPU，则使用GPU，接下来的操作同训练
#                 val_label = val_label.to(device)
#                 mask = val_input['attention_mask'].to(device)
#                 input_id = val_input['input_ids'].squeeze(1).to(device)
#
#                 output = model(input_id, mask)
#
#                 batch_loss = criterion(output, val_label)
#                 total_loss_val += batch_loss.item()
#
#                 acc = (output.argmax(dim=1) == val_label).sum().item()
#                 total_acc_val += acc
#
#         print(
#             f'''Epochs: {epoch_num + 1}
#               | Train Loss: {total_loss_train / len(train_data): .3f}
#               | Train Accuracy: {total_acc_train / len(train_data): .3f}
#               | Val Loss: {total_loss_val / len(val_data): .3f}
#               | Val Accuracy: {total_acc_val / len(val_data): .3f}''')


def train(model, train_data, val_data, learning_rate, epochs):
    # 1. 初始化记录列表
    history = []

    train, val = Dataset(train_data), Dataset(val_data)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True, num_workers=2)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        # --- 训练循环 ---
        model.train()  # 确保进入训练模式
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # --- 验证循环 ---
        total_acc_val = 0
        total_loss_val = 0
        model.eval()  # 确保进入评估模式

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        # --- 2. 计算本轮指标 ---
        epoch_train_loss = total_loss_train / len(train_data)
        epoch_train_acc = total_acc_train / len(train_data)
        epoch_val_loss = total_loss_val / len(val_data)
        epoch_val_acc = total_acc_val / len(val_data)

        # --- 3. 记录到列表 ---
        history.append({
            'epoch': epoch_num + 1,
            'train_loss': epoch_train_loss,
            'train_acc': epoch_train_acc,
            'val_loss': epoch_val_loss,
            'val_acc': epoch_val_acc
        })

        # --- 4. 实时保存到 CSV (防止程序崩溃丢失数据) ---
        df_history = pd.DataFrame(history)
        df_history.to_csv('training_log.csv', index=False)

        print(
            f'''Epochs: {epoch_num + 1}
              | Train Loss: {epoch_train_loss: .3f} | Train Accuracy: {epoch_train_acc: .3f}
              | Val Loss: {epoch_val_loss: .3f} | Val Accuracy: {epoch_val_acc: .3f}''')

    # --- 5. 训练结束后，画图展示 ---
    plot_history(df_history)

    return df_history


# 辅助函数：画图
def plot_history(df):
    plt.figure(figsize=(12, 5))

    # 画 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 画 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train_acc'], label='Train Acc')
    plt.plot(df['epoch'], df['val_acc'], label='Val Acc')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('training_result.png')  # 保存图片
    plt.show()

def evaluate(model, test_data):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


if __name__ == '__main__':
    df = pd.read_csv('./resources/cnews.train.csv', header=None, sep='\t')
    df.columns = ['category', 'text']
    print(df.head())

    np.random.seed(112)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8 * len(df)), int(.9 * len(df))])

    print(len(df_train), len(df_val), len(df_test))

    EPOCHS = 3
    model = BertClassifier()
    LR = 1e-6

    # 在调用 train() 之前，加上这两行代码检查数据
    # print("训练集标签情况:", df_train['category'].unique())
    # print("验证集标签情况:", df_val['category'].unique())

    train(model, df_train, df_val, LR, EPOCHS)
    evaluate(model, df_test)
    
    # 训练和评估流程结束后，保存模型权重到文件
    torch.save(model.state_dict(), 'bert_classifier.pth')
    print("模型已保存为 bert_classifier.pth")