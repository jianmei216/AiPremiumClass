"""
2. 加载处理后文本构建词典、定义模型、训练、评估、测试。
"""
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


def build_vocab(comments_data):
    """
    构建词汇表
    :param comments_data: 评论数据
    :return: 词汇表字典
    """
    vocab = {}
    vocab['PAD'] = 0  # 填充符
    vocab['UNK'] = 1  # 未知词
    for _, comment in comments_data:
        # 使用jieba分词
        words = list(jieba.cut(comment.strip()))
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)  # 为新词分配索引
    return vocab

class CommentsClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(CommentsClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # padding_idx=0
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        embedded = self.embedding(input_ids)  #embedded shape: (batch_size, seq_len, embedding_dim)
        output, (hidden, _) = self.rnn(embedded) # output shape: (batch_size, seq_len, hidden_size)
        output = self.fc(output[:, -1, :])  # 取最后一个时间步的输出 shape: (batch_size, num_classes)
        return output

if __name__ == '__main__':
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载评论数据
    comment_file = 'cleaned_DMSC.csv'
    df = pd.read_csv(comment_file)
    comments_data = df.values.tolist() # 将DataFrame转换为列表
    # 划分训练集和测试集
    train_data, test_data = train_test_split(comments_data, test_size=0.2, random_state=42)

    # 构建词汇表
    # 如果已经有词汇表文件，可以直接加载
    try:
        vocab = torch.load('comments_vocab.pth')
    except:
        vocab = build_vocab(comments_data)
        torch.save(vocab, 'comments_vocab.pth')
    print('词汇表大小:', len(vocab))

    # 自定义训练数据批次加载数据处理函数
    # collate_fn函数用于将每个batch的数据转换为tensor
    def collate_fn(batch):
        comments,labels = [],[]  # 分别存储评论和标签
        for label, comment in batch:
            comments.append(torch.tensor([vocab.get(word, vocab['UNK']) for word in jieba.cut(comment)]))
            labels.append(label)
        
        # 将评论和标签转换为tensor
        commt = pad_sequence(comments, batch_first=True, padding_value=vocab['PAD']) # 填充为相同长度
        labels = torch.tensor(labels)
        return commt, labels

    # 通过Dataset构建DataLoader
    train_dataloader = DataLoader(train_data,batch_size=32, shuffle=True,collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_size = 128
    num_classes = 2
    # 构建模型
    model = CommentsClassifier(vocab_size, embedding_dim, hidden_size, num_classes).to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 模型训练
    num_epochs = 5
    for epoch in range(num_epochs):
        for i, (commt, labels) in enumerate(train_dataloader):
            commt = commt.to(device) # 评论
            labels = labels.to(device) # 标签
            # 前向传播
            outputs = model(commt)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 打印训练信息
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
    # 保存模型 & 词典
    torch.save(model.state_dict(), 'comments_classifier.pth') #权重和偏置
    #torch.save(vocab, 'comments_vocab.pth')

    #模型评估
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for cmt, lbl in test_dataloader:
            cmt = cmt.to(device)
            lbl = lbl.to(device)
            outputs = model(cmt)
            _, predicted = torch.max(outputs.data, 1)
            total += lbl.size(0)
            correct += (predicted == lbl).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

    # 测试模型
    texts_test = ["我喜欢这部电影", "太难看了"]
    text_test_index = []
    for text in texts_test:
        idx_seq = [vocab.get(word,vocab['UNK']) for word in jieba.cut(text)]
        text_test_index.append(idx_seq)
    # 填充序列
    text_test_index = pad_sequence([torch.tensor(idx_seq) for idx_seq in text_test_index], batch_first=True, padding_value=vocab['PAD'])
    # 推理
    with torch.no_grad():
        model.eval()
        text_test_index = text_test_index.to(device)
        logits = model(text_test_index)
        _, predicted = torch.max(logits, 1)
    # 打印预测结果
    for text, pred in zip(texts_test, predicted):
        sentiment = '正面' if pred.item() == 1 else '负面'
        print(f'评论: "{text}" 的预测情感为: {sentiment}')


# Epoch [5/5], Step [82180/82182], Loss: 0.1747
# Accuracy of the model on the test set: 90.40%
# 评论: "我喜欢这部电影" 的预测情感为: 正面
# 评论: "太难看了" 的预测情感为: 负面