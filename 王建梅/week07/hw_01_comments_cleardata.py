"""
1. 使用豆瓣电影评论数据完成文本分类处理：文本预处理，加载、构建词典。（评论得分1～2,表示positive取值：1，评论得分4～5代表negative取值：0）
https://www.kaggle.com/datasets/utmhikari/doubanmovieshortcomments
"""
import json
import re
from typing import final
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import jieba
import pandas as pd

def clear_commentsByCSVFile(file_name):
    """
    清洗评论数据，过滤掉不符合要求的行
    :param file_name: CSV文件名
    :return: 清洗后的CSV文件名
    :只保留评分1，2, 4, 5的行
    :只保留评论不为空，且长度大于2的行
    """
    # 读取CSV文件,只保留评论和评分两列
    df = pd.read_csv(file_name, usecols=['Comment', 'Star'])
    print('原始数据行数:', len(df))
    # 只保留star in [1, 2, 4, 5]的行
    df = df[df['Star'].isin([1, 2, 4, 5])]
    print('筛选后数据行数:', len(df))
    # 过滤掉评论为空，star列为NaN，评论长度小于2的行,并重置索引
    df = df.dropna(subset=['Comment'])
    df = df[df['Comment'].str.len() > 2]
    # 将评论得分1～2,表示positive取值：1，评论得分4～5代表negative取值：0
    df['Star'] = df['Star'].apply(lambda x: 1 if x >= 4 else 0)
    print('清洗后数据行数:', len(df))
    # 保存清洗后的数据
    final_file_name = 'cleaned_' + file_name
    df.to_csv(final_file_name,index=False, encoding='utf-8')
    return final_file_name

# 构建词汇表
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

if __name__ == '__main__':
    # 设备
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 清洗语料(使用pandas读取csv文件)
    file_name = 'DMSC.csv'
    final_file_name = clear_commentsByCSVFile(file_name)
    
    # 加载清洗后的CSV文件
    df = pd.read_csv(final_file_name)
    # 将DataFrame转换为列表
    comments_data = df.values.tolist()
    # 构建词汇表
    vocab = build_vocab(comments_data)
    torch.save(vocab, 'comments_vocab.pth')  # 保存词汇表
    # 输出词汇表大小
    print('词汇表大小:', len(vocab))
    
    
