作业：
1. 使用nano-gpt5.0训练文本语料，提升内容生成可读性。
2. 使用量化方式加载文本生成大模型并进行推理
3. 使用vllm加载LLM大模型并测试文本生成。

（模型可以从huggingface或modelscope中寻找）

作业总结：

1. 英文语料loss效果好于中文，(英文：loss 4.4  - > 2.4,中文_v1 loss 8.6785 -> 2.90,中文_v2 loss 8.5819 -> )，中文_v1 是切好token的红楼梦语料HLM_utf8.txt,中文_v2是没有切分token的语料hlm.txt，从结果上观察，两个语料的loss中文_v1的值更低一些，但是推理结果上看,v2的更顺畅一些，虽然都不通顺，但优于上次的gpt2.0)，此处由于语料并不丰富，因此其质量会造成推理结果差异较大。
2. bitsandbytes量化方法，4位整型精度，bitsandbytes主要是为 NVIDIA GPU 设计的量化工具，核心优化（如 4-bit/8-bit 量化）严重依赖 NVIDIA 的 CUDA 加速，在 CPU 环境下无法启用这些低精度量化功能。如果强制在 CPU 上使用，会出现类似 `CUDA not available` 的错误，或自动回退到 FP32 精度（失去量化意义）。
   本次实验选择三个huggingface不受限的小模型尝试(facebook/opt-1.3b，EleutherAI/gpt-neo-1.3B,bigscience/bloom-1b7)，分别对英文文本，中文文本进行有量化&无量化推理，结果显示不同的生成模型效果区别很大，但是否量化对推理结果影响不大。这几个模型对常见的英文故事"Once upon a time,"不能生成很好的后续，而中文方面，只有bigscience/bloom-1b7模型对经典的故事开头"从前有座山",进行了正常的故事描述。结论是：模型量化为4位后，推理效果并未有明显降低。
3. vllm是一个企业级高性能推理框架，内置 INT8、FP16 等量化支持，结合 PagedAttention 技术大幅提升吞吐量，适合高并发场景，但量化功能相对简化。本次实验在kaggle中进行，使用 GPU，`建议tensor_parallel_size`=1，若遇到显存不足，降低 `gpu_memory_utilization`（如 0.8），vllm 擅长批量处理，可增加 prompts 数量提高效率，代码中使用的vllm 会自动优化推理过程（如使用 PagedAttention 技术），相比传统的 transformers 库能提供更高的吞吐量和更低的延迟。
   vllm 的 PagedAttention 优化需要 GPU 计算能力 ≥ 7.0（如 NVIDIA T4、V100、A100、RTX 20 系列及以上）<7.0 （如P100 的 CC为6.0）不能使用。
