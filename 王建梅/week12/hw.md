作业内容：

1. 利用上周NER模型训练任务代码，复现课堂案例中：动态学习率、混合精度、DDP训练实现。
2. 利用课堂案例，实现分布式DDP模型训练。存盘后加载实现推理。

总结：

1. 在week11的NER模型训练代码基础上，修改了训练参数，新增了以下参数
   local_rank=rank,   # 当前进程 RANK，DDP优化必要参数
   fp16=True,               # 使用混合精度，当前使用GPU T4，支持FP16
   lr_scheduler_type='linear',  # 动态学习率
   warmup_steps=100,        # 预热步数
   ddp_find_unused_parameters=False  # 优化DDP性能

   三轮训练后，loss收敛快速正常，准确率也和week11相似，loss无异常波动或不收敛，表示不需要关闭 FP16，但训练时间比week11多了4分钟。
2. kaggle中训练时间依然比较久，没有使用pytorch原生API实现DDP训练流程，使用了主流的HuggingFace Trainer框架实现DDP模型训练，随机选择文本进行推理，流程没问题，但推理效果不理想。
