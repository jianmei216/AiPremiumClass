"""
使用nano-gpt5.0训练文本语料，提升内容生成可读性。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def get_batch(split):
    # 选择训练或验证数据集
    data = train_data if split == 'train' else val_data

    # 动态从数据集中选择位置索引
    ix = torch.randint(len(data) - block_size, (batch_size,)) # [0,103846]随机生成位置索引，向后截取block_size字符训练
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x.to(device),y.to(device)

class Block(nn.Module):

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, n_embd, head_size, dropout)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # 残差连接
        x = x + self.ffwd(self.ln2(x)) # 残差连接
        return x

class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, n_embd, head_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_embd, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_embd, bias=False)
        self.query = nn.Linear(n_embd, head_embd, bias=False)
        self.value = nn.Linear(n_embd, head_embd, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_x):
        B, T, C = input_x.shape
        k = self.key(input_x)   # (B, T, head_size)
        q = self.query(input_x) # (B, T, head_size)
        v = self.value(input_x) # (B, T, 16)

        wei = q @ k.transpose(-2,-1) * C ** -0.5

        T = wei.shape[-1]
        tril = torch.tril(torch.ones(T, T, device=wei.device))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = wei.softmax(dim=-1)
        wei = self.dropout(wei)
        
        out = wei @ v
        return out

        ##################### flash attention 2 #########################

        # FlashAttention2 只支持bfloat16或float16类型的张量
        # q = q.to(torch.bfloat16)
        # k = k.to(torch.bfloat16)
        # v = v.to(torch.bfloat16) # 或调用 v.bfloat16()

        # with sdpa_kernel(backends=SDPBackend.FLASH_ATTENTION):
        #     attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # return attn_output


class BingramLanguageModel(nn.Module):
    
    def __init__(self, block_size, vocab_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        # 每个token直接输出的logits值作为下一个token的映射
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx和target都是维度为 (B,T) 的整型tensor
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device), ) # (T,C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx指当前语料集(B,T)中的索引
        for _ in range(max_new_tokens):
            # 限定索引列的取值范围
            idx_cond = idx[:, -block_size:]
            # 推理
            logits, loss = self(idx_cond)
            # 只提取最后一个时间步的结果
            logits = logits[:, -1, :]  # (B,C)
            # 通过softmax转换为概率值
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # 随机采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # 把采样的索引追加在当前解码序列末尾
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

if __name__ == '__main__':
 
    # 模型训练数据集
    block_size = 8
    batch_size = 32
    max_iter = 5000
    learn_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd = 32
    eval_interval = 500
    eval_iters = 200
    head_size = 8
    num_layers = 4
    dropout = 0.1

    
    with open('hlm.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # 字典、编码器(函数)、解码器(函数)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i,ch in enumerate(chars)}  #str_to_index
    itos = {i:ch for i,ch in enumerate(chars)}  #index_to_str

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # 文本转换token index
    data = torch.tensor(encode(text), dtype=torch.long)

    # 拆分数据集
    n = int(len(data) * .9)
    train_data = data[:n]
    val_data = data[n:]

    # 模型训练
    model = BingramLanguageModel(block_size, vocab_size, n_embd, head_size, num_layers, dropout)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    for iter in range(max_iter):

        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # 批次样本
        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # 模型生成
    idx = torch.zeros((1,1), dtype=torch.long, device=device)
    print(decode(model.generate(idx, max_new_tokens=500)[0].tolist())) 

    
# ############# 训练推理结果 #########################
# --------- 英文语料 input.txt
# step 0: train loss 4.3941, val loss 4.3863
# step 500: train loss 2.4387, val loss 2.4393
# step 1000: train loss 2.2838, val loss 2.2902
# step 1500: train loss 2.2162, val loss 2.2240
# step 2000: train loss 2.1540, val loss 2.1745
# step 2500: train loss 2.1152, val loss 2.1617
# step 3000: train loss 2.0797, val loss 2.1129
# step 3500: train loss 2.0537, val loss 2.1182
# step 4000: train loss 2.0385, val loss 2.1003
# step 4500: train loss 2.0181, val loss 2.0857

# Wetcosts,
# As:
# And we?

# Yhave sord aberves:
# And ave king inly our
# And we?

# Yhave sord aberves:
# And ave king inly our
# Lorrat whath vosiculy teat gicuse.

# --------    中文语料 HLM_utf8.txt 红楼梦
# step 0: train loss 8.6844, val loss 8.6785
# step 500: train loss 3.6547, val loss 3.5666
# step 1000: train loss 3.3648, val loss 3.2869
# step 1500: train loss 3.2360, val loss 3.1521
# step 2000: train loss 3.2032, val loss 3.0847
# step 2500: train loss 3.1294, val loss 3.0338
# step 3000: train loss 3.0840, val loss 3.0029
# step 3500: train loss 3.0300, val loss 2.9701
# step 4000: train loss 3.0407, val loss 2.9501
# step 4500: train loss 2.9976, val loss 2.9010

#  自贾环 荔个 ， 补已 了 半包 ‘ 洛么 和 老县 的 掩院 。 ” 素日 听见 的 敛 的 ， 见 报疯 ‘ 东西 许多 ， 老 便 向妈 老嬷 见 坐 ， 哪里 忽到 夜戚 ， 又 怕 活枝 ， 又 一句 并 再 熬 该 。 不倒 来 坐 ， 是 编情 米间 的 喜衫凶食 ， 只见 说 心个 站 ， 无声 比 她 盆 宝且 着 。 ” 宝爷 听 了 不曾 说 了 岳玉 被 ， 盖 孙 了 就 你 身是  什样 花玉 看 看成 此时 ， 薛姨娘 都 不没 了 白成 ， 也 来有 怕年 些 地中 命何 。 其 便 齐 一声 的 ， 雨村 不过 咕夫 了 过 。


#  带 宝钗 道 谁 有 收 那 ， 一势 嵌蔷 唱府 京 ， 乃泪佛 只是 字 。 再 知丫头 给 省虚下 ， 今天家 道 ： “ 求袭 虽 又 仍 了 大法 ， 忽为 大家 禀陪 笑 这些 。 一两样 地史 无事 一回 什么 人 儿子 筝心 怎么 容 了 迎春 ， 我 又 各人 都 可 找家 了 了 一把 呢 一官 的 ， 炮琅 我 一 那个 各中 喝 女子 的 梦牌 。 或来 别叶 听 了 银子 茶居 ， “ 知

# ------  中文语料 hlm.txt
# step 0: train loss 8.5859, val loss 8.5819
# step 500: train loss 5.5023, val loss 5.4232
# step 1000: train loss 5.1169, val loss 5.0449
# step 1500: train loss 4.9527, val loss 4.8945
# step 2000: train loss 4.8067, val loss 4.7868
# step 2500: train loss 4.7071, val loss 4.6958
# step 3000: train loss 4.6948, val loss 4.6191
# step 3500: train loss 4.6453, val loss 4.5997
# step 4000: train loss 4.6289, val loss 4.6110
# step 4500: train loss 4.5538, val loss 4.5520

# 千家留了阁心，别人进他口府请炕璃净，揭然用，且母来想了。及向次，自伤意混有么，倒倒是个命的本早唾花，打发像会小意思的正说道二书说：“高尽箭门，带雨丫头升的；这么大 事自然说不知先来，一边去请厉。”宝玉道：“后家的浑贾蓉开着，恰身既来报呈，可咱们叫：“你要惟躲打了。”又束阕人听了，方才真“小饽住，先恐系钮所以在玩的没有个要出什么， 看见成白邻收，见外做性不好手的说，抽来了找唬冷笑吗，便脑红了，化上，道至小子───今忙大天，只见，宁二爷耳帘旧垂，盥着薛二府却一点子的，去，早说道：“他怎么奶奶这人，颤山上罢，如何何东西！”于是孔头，至赵姨奶奶奶奶给合我？”宝二叔叔听，便禅算移真奇清壮烟四眼生儿，谁人坐着罢捆头找，合贵
# 正翻头房子了一副，他亲江体统全眈茶呢。宝玉回了一个人，知唤姑娘到底遇的？再搥缕的心里，点事来闹着，就算定人。一路头方也敢说：“浊俊吃些门更招得。其采尘台喜戏，你就 有今情的？
# 如今天安错饵银了。”贾赦也道：“没不用者玩，倒到我自看说，心里看算用心的辈子又冷道：“当日这个生么理儿，我还有什么？”
# 宝钗听复点手，死了，反全他们歇着，这一算定不嚷？”薛蟠便又捧将不知怕来。雪雁最大家替不越发