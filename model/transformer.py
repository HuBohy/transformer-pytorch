import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# pylint: disable=arguments-differ

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=12):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v, cache=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, t):  # pylint: disable=arguments-differ
        y = self.self_attention_norm(x)
        t = self.self_attention_norm(t)

        y = self.self_attention(y, t, t)

        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x

class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers):
        super(Encoder, self).__init__()

        encoders = [EncoderLayer(hidden_size, filter_size, dropout_rate)
                    for _ in range(n_layers)]
        self.layer = nn.ModuleList(encoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, inputs, targets):
        encoder_output = inputs
        for enc_layer in self.layer:
            encoder_output = enc_layer(encoder_output, targets)
        return (encoder_output)

class Bottleneck(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers, fusion_layer=0) -> None:
        super().__init__()

        encoders = [EncoderLayer(hidden_size, filter_size, dropout_rate)
                    for _ in range(fusion_layer)]
        
        bottlenecks = [EncoderLayer(hidden_size, filter_size, dropout_rate)
                    for _ in range(n_layers-fusion_layer)]

        self.video_layer = nn.ModuleList(encoders)
        self.audio_layer = nn.ModuleList(encoders)

        self.video_bottleneck_layer = nn.ModuleList(bottlenecks)
        self.audio_bottleneck_layer = nn.ModuleList(bottlenecks)

        self.video_last_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.audio_last_norm = nn.LayerNorm(hidden_size, eps=1e-6)


    def forward(self, inputs, targets=None, modality_thetas=[0, 0]):
        encoder_output = inputs
        
        video_output = inputs[:, :modality_thetas[0]]
        audio_output = inputs[:, modality_thetas[1]:]
        bottleneck_output = inputs[:, modality_thetas[0]:modality_thetas[1]]

        for vid_layer, aud_layer in zip(self.video_layer, self.audio_layer):
            video_output = vid_layer(video_output, video_output)
            audio_output = aud_layer(audio_output, audio_output)

        for vid_layer, aud_layer in zip(self.video_bottleneck_layer, self.audio_bottleneck_layer):
            video_output = torch.cat((video_output, bottleneck_output), dim=1)
            audio_output = torch.cat((bottleneck_output, audio_output), dim=1)

            video_output = vid_layer(video_output, video_output)
            audio_output = aud_layer(audio_output, audio_output)
            
            bottleneck_output = (video_output[:, modality_thetas[0]:] + audio_output[:, :modality_thetas[1]-modality_thetas[0]])/2

            video_output = video_output[:, :modality_thetas[0]]
            audio_output = audio_output[:, modality_thetas[1]-modality_thetas[0]:]
            
        encoder_output = torch.cat((video_output, bottleneck_output, audio_output), dim=1)

        return (encoder_output)


class CrossModalTransformer(nn.Module):
    def __init__(self,
                 n_layers=12,
                 hidden_size=768,
                 filter_size=3072,
                 dropout_rate = 0.1
                ):
        super(CrossModalTransformer, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.hidden_size = hidden_size
        # self.emb_scale = hidden_size ** 0.5

        self.input_embeddings = Embeddings()
        self.target_embeddings = Embeddings()
        self.encoder = Encoder(hidden_size, filter_size,
                                dropout_rate, n_layers).cuda()

        self.layernorm = nn.LayerNorm(hidden_size, 1e-12).cuda()
        self.pooler = Pooler(hidden_size)

    def forward(self, inputs, targets=None):
        encoded_input = inputs
        encoded_input += self.input_embeddings.position_embedddings[:, :encoded_input.shape[1]]
        encoded_input = self.input_embeddings.dropout(encoded_input)

        encoded_target = targets
        encoded_targets += self.target_embeddings.position_embedddings[:, :encoded_target.shape[1]]
        encoded_targets = self.target_embeddings.dropout(encoded_target)

        enc_output = self.encoder(encoded_input, encoded_target)[0]

        return enc_output

    # def get_position_encoding(self, x):
    #     max_length = x.size()[1]
    #     position = torch.arange(max_length, dtype=torch.float32,
    #                             device=x.device)
    #     scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0).to(x.device)
    #     signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
    #                        dim=1)
    #     signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
    #     signal = signal.view(1, max_length, self.hidden_size)
    #     return signal

class Pooler(nn.Module):
    def __init__(self, hidden_size=768) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, inputs, idx = 0):
        first_token_tensor = inputs[:, idx]
        pooled_output = self.dense(first_token_tensor)
        pooled_output =self.activation(pooled_output)
        return pooled_output

class Embeddings(nn.Conv3d):
    def __init__(
            self,
            hidden_size = 768,
            kernel_size = (4, 16, 16),
            dropout_rate=0.1
        ) -> None:
        super().__init__(
            3,
            hidden_size,
            kernel_size,
            stride=kernel_size,
            padding='valid',
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        self.position_embedddings = nn.Parameter(torch.zeros((1, 3137, hidden_size)))
        self.cls_token = nn.Parameter(torch.zeros((1, 1, hidden_size)))
        self.dropout = nn.Dropout(dropout_rate)

class BottleneckTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()