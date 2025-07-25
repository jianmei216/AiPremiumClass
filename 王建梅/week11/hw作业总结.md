1. 语料中已经将文本切分成词，且已经将对应的标签索引标注好，因此dataset只需要将input_ids,labels处理好即可
2. 使用tokenizer.convert_tokens_to_ids方法将已切分好的tokens转换成token索引
3. 超过512长度的语料截断
4. 三轮训练后，3个类别精准率和召回率均高于90%，其中Loc的f1成绩最好达到95%
5. 使用model(**input_data) 和 trainer.predict([input_data])预测，input_data中没有labels数据，所以trainer.predict预测时，result中label_ids=None，应使用predictions计算
