* 参考课堂案例，使用指定的数据集，编写代码实现ner模型训练和推理。
  https://huggingface.co/datasets/doushabao4766/msra_ner_k_V3
* 完成预测结果的实体抽取。
  输入：“双方确定了今后发展中美关系的指导方针。”
  输出：[{"entity":"ORG","content":"中"},{"entity":"ORG","content":"美"}]
* 整理Dataset、Trainer、TrainingArgument、DataCollator、Evaluate 知识点，总结文档
