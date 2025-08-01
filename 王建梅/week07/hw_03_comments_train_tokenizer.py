"""
3. 尝试不同分词工具进行文本分词，观察模型训练结果。
"""
import os
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# 原生分词方法，即将文本按字符切分
def build_vocab_native(comments_data):
    """
    构建词汇表
    :param comments_data: 评论数据
    :return: 词汇表字典
    """
    vocab = {}
    vocab['PAD'] = 0  # 填充符
    vocab['UNK'] = 1  # 未知词
    for _, comment in comments_data:
        words = list(comment)  # 按字符切分
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)  # 为新词分配索引
    torch.save(vocab, 'comments_vocab_3_native.pth')  # 保存词汇表-原生分词
    return vocab
def build_vocab_jieba(comments_data):
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
        words = list(jieba.cut(comment))
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)  # 为新词分配索引
    torch.save(vocab, 'comments_vocab_3_jieba.pth')  # 保存词汇表-jieba分词
    return vocab

def get_vocab(vocab_type,comments_data):
    """
    获取词汇表
    :param vocab_type: 分词工具类型
    :return: 词汇表字典
    """
    if vocab_type == 'jieba':
        # 尝试加载jieba分词的词汇表
        if os.path.exists('comments_vocab_3_jieba.pth'):
            return torch.load('comments_vocab_3_jieba.pth')
        return build_vocab_jieba(comments_data)  # 如果不存在，则构建新的词汇表
    else:
        # 尝试加载原生分词的词汇表
        if os.path.exists('comments_vocab_3_native.pth'):
            return torch.load('comments_vocab_3_native.pth')
        return build_vocab_native(comments_data)

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
    # 构建词汇表
    #build_vocab_native(comments_data)  # 构建原生分词的词汇表
    #build_vocab_jieba(comments_data)  # 构建jieba分词
    # 划分训练集和测试集
    train_data, test_data = train_test_split(comments_data, test_size=0.2, random_state=42)

    # 加载词汇表
    vocab_jieba = get_vocab('jieba',comments_data)
    print('jieba词汇表大小:', len(vocab_jieba))
    vocab_native = get_vocab('native',comments_data)
    print('原生分词词汇表大小:', len(vocab_native))

    # 自定义训练数据批次加载数据处理函数
    # collate_fn函数用于将每个batch的数据转换为tensor
    def collate_fn(batch,vocab_type):
        comments,labels = [],[]  # 分别存储评论和标签
        for label, comment in batch:
            if vocab_type == 'jieba':
                # 使用jieba分词
                words = list(jieba.cut(comment))
                vocab = vocab_jieba
            else:
                # 原生分词方法
                words = list(comment.strip())
                vocab = vocab_native
            comments.append(torch.tensor([vocab.get(word, vocab['UNK']) for word in words]))
            labels.append(label)
        
        # 将评论和标签转换为tensor
        commt = pad_sequence(comments, batch_first=True, padding_value=vocab['PAD']) # 填充为相同长度
        labels = torch.tensor(labels)
        return commt, labels

    # 通过Dataset构建DataLoader
    # train_dataloader_jieba = DataLoader(train_data,batch_size=512, shuffle=True,collate_fn=collate_fn(vocab_type='jieba'))
    # test_dataloader_jieba = DataLoader(test_data, batch_size=512, shuffle=False, collate_fn=collate_fn(vocab_type='jieba'))
    # train_dataloader_native = DataLoader(train_data,batch_size=512, shuffle=True,collate_fn=collate_fn(vocab_type='native'))
    # test_dataloader_native = DataLoader(test_data, batch_size=512, shuffle=False, collate_fn=collate_fn(vocab_type='native'))
    
    embedding_dim = 100
    hidden_size = 128
    num_classes = 2
    
    def train_model(vocab_type):
        print(f"Training model with {vocab_type} vocabulary...")
        vocab_type = 'jieba' if vocab_type == 'jieba' else 'native'
        train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=lambda x: collate_fn(x, vocab_type=vocab_type))
        if vocab_type == 'jieba':
            vocab = vocab_jieba  
        else:
            vocab = vocab_native

        # 构建模型
        model = CommentsClassifier(len(vocab), embedding_dim, hidden_size, num_classes).to(device)
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
        # 保存模型
        torch.save(model.state_dict(), 'comments_classifier_3_'+vocab_type+'.pth') #权重和偏置

    def eval_model(vocab_type):
        vocab_type = 'jieba' if vocab_type == 'jieba' else 'native'
        test_dataloader = DataLoader(test_data, batch_size=512, shuffle=False, collate_fn=lambda x: collate_fn(x, vocab_type=vocab_type))
        if vocab_type == 'jieba':
            vocab = vocab_jieba
        else:
            vocab = vocab_native

        # 加载模型
        model = CommentsClassifier(len(vocab), embedding_dim, hidden_size, num_classes).to(device)
        model.load_state_dict(torch.load('comments_classifier_3_'+vocab_type+'.pth'))
        
        # 模型评估
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for cmt, lbl in test_dataloader:
                cmt = cmt.to(device)
                lbl = lbl.to(device)
                outputs = model(cmt)
                _, predicted = torch.max(outputs.data, 1) # 获取预测结果,返回每个样本的最大概率值和对应的类别索引
                total += lbl.size(0)
                correct += (predicted == lbl).sum().item()
            accuracy = 100 * correct / total
            print(f'Accuracy of the model on the {vocab_type} test set: {accuracy:.2f}%')

    # 模型训练 & 评估
    train_model('jieba')
    eval_model('jieba')

    train_model('native')
    eval_model('native')

    # jieba词汇表大小: 287571
    # 原生分词词汇表大小: 9311
    # Training model with jieba vocabulary...
    # Building prefix dict from the default dictionary ...
    # Loading model from cache C:\Users\jianm\AppData\Local\Temp\jieba.cache       
    # Loading model cost 0.503 seconds.
    # Prefix dict has been built successfully.
    # Epoch [1/2], Step [1000/2569], Loss: 0.5103
    # Epoch [1/2], Step [2000/2569], Loss: 0.5246
    # Epoch [2/2], Step [1000/2569], Loss: 0.5351
    # Epoch [2/2], Step [2000/2569], Loss: 0.5642
    # Accuracy of the model on the jieba test set: 77.66%
    # Training model with native vocabulary...
    # Epoch [1/2], Step [1000/2569], Loss: 0.5310
    # Epoch [1/2], Step [2000/2569], Loss: 0.5571
    # Epoch [2/2], Step [1000/2569], Loss: 0.3049
    # Epoch [2/2], Step [2000/2569], Loss: 0.2019
    # Accuracy of the model on the native test set: 90.65%