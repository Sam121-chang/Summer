"""
使用类型注解和最佳实践的简化版Transformer实现
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """用于非循环序列的正弦位置编码。

    参数:
        d_model: 嵌入维度
        max_len: 最大序列长度
        dropout: 丢弃概率
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦编码
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦编码
        self.register_buffer('pe', pe)  # 将位置编码注册为缓冲区，不视为模型参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将位置编码添加到输入张量。

        参数:
            x: 形状为[batch_size, seq_len, embedding_dim]的输入张量

        返回:
            添加了位置编码的张量
        """
        if x.dim() != 3:
            raise ValueError(f"期望输入维度为3，得到{x.dim()}")
        x = x + self.pe[:x.size(1)].transpose(0, 1)  # 将位置编码添加到词嵌入
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头自注意力机制。

    参数:
        d_model: 嵌入维度
        num_heads: 注意力头数
        dropout: 丢弃概率
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"嵌入维度({d_model})必须能被注意力头数({num_heads})整除"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 线性投影层
        self.q_linear = nn.Linear(d_model, d_model)  # 查询矩阵投影
        self.k_linear = nn.Linear(d_model, d_model)  # 键矩阵投影
        self.v_linear = nn.Linear(d_model, d_model)  # 值矩阵投影

        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(d_model, d_model)  # 输出线性层

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算多头注意力。

        参数:
            query: 查询张量 [batch_size, q_len, d_model]
            key: 键张量 [batch_size, k_len, d_model]
            value: 值张量 [batch_size, v_len, d_model]
            mask: 可选掩码张量 [batch_size, 1, q_len, k_len]

        返回:
            attention_output: 输出张量 [batch_size, q_len, d_model]
            attention_weights: 注意力权重 [batch_size, num_heads, q_len, k_len]
        """
        batch_size = query.size(0)

        # 线性投影
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # 分割为多个注意力头
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用掩码(如果提供)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 将注意力权重应用到值上
        attention_output = torch.matmul(attention_weights, V)

        # 连接多个注意力头的输出
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        # 最终线性投影
        output = self.out_linear(attention_output)

        return output, attention_weights


class PositionWiseFFN(nn.Module):
    """位置前馈神经网络。

    参数:
        d_model: 嵌入维度
        d_ff: 隐藏层维度
        dropout: 丢弃概率
    """

    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 第一个线性层
        self.linear2 = nn.Linear(d_ff, d_model)  # 第二个线性层
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 使用GELU激活函数，比ReLU更现代

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播通过前馈网络。

        参数:
            x: 输入张量 [batch_size, seq_len, d_model]

        返回:
            输出张量 [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """具有残差连接的单个Transformer编码器层。

    参数:
        d_model: 嵌入维度
        num_heads: 注意力头数
        d_ff: 前馈网络隐藏维度
        dropout: 丢弃概率
    """

    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 8,
            d_ff: int = 2048,
            dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)  # 自注意力层
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)  # 前馈网络

        # 层归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播通过编码器层。

        参数:
            src: 输入序列 [batch_size, src_len, d_model]
            src_mask: 可选掩码 [batch_size, 1, src_len, src_len]

        返回:
            转换后的序列 [batch_size, src_len, d_model]
        """
        # 自注意力与残差连接
        attn_output, _ = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(attn_output)  # 残差连接
        src = self.norm1(src)  # 层归一化

        # 前馈网络与残差连接
        ffn_output = self.ffn(src)
        src = src + self.dropout2(ffn_output)  # 残差连接
        src = self.norm2(src)  # 层归一化

        return src


class SimplifiedTransformer(nn.Module):
    """用于序列处理的简化版Transformer模型。

    参数:
        vocab_size: 输入词汇表大小
        d_model: 嵌入维度
        num_layers: 编码器层数
        num_heads: 注意力头数
        d_ff: 前馈网络隐藏维度
        dropout: 丢弃概率
        max_len: 最大序列长度
    """

    def __init__(
            self,
            vocab_size: int,
            d_model: int = 512,
            num_layers: int = 6,
            num_heads: int = 8,
            d_ff: int = 2048,
            dropout: float = 0.1,
            max_len: int = 5000
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)  # 词嵌入层
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)  # 位置编码器

        # 创建多层编码器
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)  # 缩放因子，用于缩放词嵌入

    def forward(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播通过Transformer。

        参数:
            src: 输入序列 [batch_size, src_len]
            src_mask: 可选掩码 [batch_size, 1, src_len, src_len]

        返回:
            输出序列 [batch_size, src_len, d_model]
        """
        # 嵌入令牌并缩放
        src = self.embedding(src) * self.scale  # 应用词嵌入并缩放
        src = self.pos_encoder(src)  # 添加位置编码

        # 依次通过每个编码器层
        for layer in self.layers:
            src = layer(src, src_mask)

        return src