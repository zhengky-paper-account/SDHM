import torch
import torch.nn as nn
import torch.nn.functional as F
from .AMA import PreNormAttention, PreNormForward, FeedForward, Attention,CrossTransformerEncoder
from mamba_ssm import Mamba
from einops import repeat
import math

# U-Net for Diffusion Models
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv1d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv1d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv1d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm1d(out_ch)
        self.bnorm2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x, t):
        # First conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend time embeddings
        time_emb = time_emb[(..., ) + (None, ) * (len(h.shape) - 2)]
        # Add time channel
        h = h + time_emb
        # Second conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Up
        return self.transform(h)

class UNetModel(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, hidden_dim=256, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # 真实的U-Net结构 - 带跳跃连接
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # Bottleneck with time embedding
        self.bottleneck = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # Decoder with proper skip connections
        self.decoder1 = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 3, padding=1),  # 2倍通道数因为有跳跃连接
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 3, padding=1),  # 2倍通道数因为有跳跃连接
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_channels, 3, padding=1),
        )
        
        # Time projection
        self.time_projection = nn.Linear(time_dim, hidden_dim)
        
    def forward(self, x, time):
        # x: [B, T, C] -> [B, C, T]
        if len(x.shape) == 3:
            x = x.transpose(1, 2)
        
        # Time embedding
        t = self.time_mlp(time)
        
        # Encoder with skip connections
        encoded1 = self.encoder1(x)  # 保存用于跳跃连接
        encoded2 = self.encoder2(encoded1)  # 保存用于跳跃连接
        
        # Add time information to bottleneck
        time_emb = self.time_projection(t)
        time_emb = time_emb.unsqueeze(-1).expand(-1, -1, encoded2.shape[-1])
        encoded2 = encoded2 + time_emb
        
        # Bottleneck
        bottleneck = self.bottleneck(encoded2)
        
        # Decoder with real skip connections
        # Up 1: bottleneck + encoded2
        decoded1 = self.decoder1(torch.cat([bottleneck, encoded2], dim=1))
        
        # Up 2: decoded1 + encoded1
        decoded2 = self.decoder2(torch.cat([decoded1, encoded1], dim=1))
        
        # [B, C, T] -> [B, T, C]
        if len(decoded2.shape) == 3:
            decoded2 = decoded2.transpose(1, 2)
            
        return decoded2

# ✅ Expert 子网络
class MLPExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ✅ 路由器
class TopKRouter(nn.Module):
    def __init__(self, embed_dim, num_experts, top_k, noisy_gating=True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.linear = nn.Linear(embed_dim, num_experts)

    def forward(self, x):
        logits = self.linear(x)  # (B*T, num_experts)
        if self.noisy_gating:
            noise = torch.randn_like(logits) / 10
            logits = logits + noise
        scores = torch.softmax(logits, dim=-1)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        return topk_scores, topk_indices, scores

# ✅ MoE Layer
class MoEBlock(nn.Module):
    def __init__(self, embed_dim, num_experts=4, top_k=2, hidden_dim=512, output_dim=None, expert_dropout=0.0):
        super().__init__()
        self.router = TopKRouter(embed_dim, num_experts, top_k, noisy_gating=True)
        self.experts = nn.ModuleList([
            MLPExpert(embed_dim, hidden_dim, output_dim or embed_dim)
            for _ in range(num_experts)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_dim = output_dim or embed_dim
        self.expert_dropout = nn.Dropout(expert_dropout) if expert_dropout > 0 else nn.Identity()
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        scores, indices, router_z = self.router(x_flat)
        output = torch.zeros(B * T, self.output_dim, device=x.device)
        self.last_router_z = router_z

        for i in range(self.top_k):
            idx = indices[:, i]
            w = scores[:, i].unsqueeze(-1)
            for eid in torch.unique(idx):
                mask = (idx == eid)
                out = self.experts[eid](x_flat[mask])
                output[mask] += w[mask] * self.expert_dropout(out)

        return self.norm(output.view(B, T, D) + x)

    def router_loss(self):
        z = self.last_router_z
        me = z.mean(dim=0)
        ce = (z ** 2).mean(dim=0)
        return (me * ce).sum() * (self.num_experts ** 2)



# 1. 高效 Linear Attention（支持 LSM 层）
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, normalize=True):
        super().__init__()
        self.heads = heads
        #self.dim_head = dim_head
        #self.inner_dim = heads * dim_head
         # 如果 dim_head 未指定，自动计算
        if dim_head is None:
            assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
            self.dim_head = dim // heads
        else:
            self.dim_head = dim_head
        self.inner_dim = heads * self.dim_head
        self.normalize = normalize

        # 三通道投影
        self.q_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.inner_dim, bias=False)

        self.out_proj = nn.Linear(self.inner_dim, dim)

    def forward(self, v, k, q):
        B, T, _ = q.shape
        
        H, D = self.heads, self.dim_head
   
        # 投影并 reshape 成多头格式
        q = self.q_proj(q).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        k = self.k_proj(k).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(v).view(B, T, H, D).transpose(1, 2)

        # 映射
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Step 1: KᵀV
        kv = torch.einsum("bhtd,bhte->bhde", k, v)  # [B, H, D, D]

        # Step 2: Q × KV
        out = torch.einsum("bhtd,bhde->bhte", q, kv)  # [B, H, T, D]

        # Step 3: normalize（与传统 Linear Attention 相同）
        if self.normalize:
            k_sum = k.sum(dim=2) + 1e-6  # [B, H, D]
            denom = torch.einsum("bhtd,bhd->bht", q, k_sum).unsqueeze(-1)
            out = out / denom

        # Reshape + 输出映射
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


#2. Mamba 封装（需安装 mamba-ssm）

class MambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = Mamba(dim)
    def forward(self, x):
        return self.block(x)
#3. Linear RNN
class LinearRNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.U = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, D = x.shape
        h = torch.zeros(B, D, device=x.device)
        outputs = []
        for t in range(T):
            h = self.W(x[:, t]) + self.U(h)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


# ✅ 封装模块：LSMBlock
# -------------------------
class LSMBlock(nn.Module):
    def __init__(self, dim, lsm_type='linear_attention'):
        super().__init__()
        if lsm_type == 'linear_attention':
            self.lsm = LinearAttention(dim)
        elif lsm_type == 'mamba':
            self.lsm = MambaBlock(dim)
        elif lsm_type == 'linear_rnn':
            self.lsm = LinearRNN(dim)
        else:
            raise ValueError(f"[LSMBlock] Unknown lsm_type: {lsm_type}")

    def forward(self, v, k, q):
        # 根据LSM类型决定如何处理参数
        if isinstance(self.lsm, LinearAttention):
            # LinearAttention需要三个参数
            return self.lsm(v, k, q)
        else:
            # MambaBlock, LinearRNN只需要一个参数
            return self.lsm(q)
    


# ✅ LSM + MoE 联合模块
class LinearMoEBlock(nn.Module):
    def __init__(self, dim, num_experts, top_k=2, hidden_dim=512, lsm_type='linear_attention'):
        super().__init__()
        self.norm_v = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.attn = LSMBlock(dim, lsm_type=lsm_type)
        self.norm_moe = nn.LayerNorm(dim)
        self.moe = MoEBlock(dim, num_experts=num_experts, top_k=top_k, hidden_dim=hidden_dim, output_dim=dim)

    def forward(self, v, k, q):
        attn_out = self.attn(self.norm_v(v), self.norm_k(k), self.norm_q(q))
        q = q + attn_out
        moe_out = self.moe(self.norm_moe(q))
        q = q + moe_out
        return q
    

# ✅ 标准 Transformer 块（用于混合）
class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, v, k, q):
        for attn, ff in self.layers:
            q = attn(v, k, q) + q
            q = ff(q) + q
        return q


class TransformerMoEBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, num_experts, top_k, dropout=0.1):
        super().__init__()
        self.attn = PreNormAttention(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ffn = MoEBlock(embed_dim=dim, num_experts=num_experts, top_k=top_k, hidden_dim=mlp_dim)
    
    def forward(self, v, k, q):
        q = self.attn(v, k, q) + q
        q = self.ffn(q) + q  # 替代原始 FFN
        return q


# ✅ 主体模块：支持 LNLN、LLNL 等结构
class HybridLinearMoEBlock(nn.Module):
    def __init__(self,  *, num_frames=50, token_len=8, save_hidden,dim, depth, structure="LNLN", 
                num_experts=4, top_k=2, hidden_dim, 
                lsm_type="linear_attention", heads=8, mlp_dim=128,
                pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.1,pos_true=True):
        """
        structure: 字符串，如 "LNLN" 表示交替使用 LinearMoEBlock 和 StandardTransformerBlock
        pos_true: 是否启用位置编码，默认为True
        """
        super().__init__()

        
        self.token_len = token_len
        self.save_hidden = save_hidden
        self.pos_true = pos_true

        if self.pos_true:
            if token_len is not None:
                self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
                self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
            else:
                 self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
                 self.extra_token = None
        else:
            self.pos_embedding = None
            self.extra_token = None

        self.dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList()
        for i in range(depth):
            block_type = structure[i % len(structure)]
            if block_type == "L":
                self.layers.append(
                    LinearMoEBlock(
                        dim=dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        hidden_dim=hidden_dim,
                        lsm_type=lsm_type  # ✅ 传入可选的 LSM 类型
                    )
                )
            else:
                self.layers.append(
                    TransformerEncoder(#小型数据集
                        dim=dim, 
                        depth=1, 
                        heads=8, 
                        dim_head=64, 
                        mlp_dim=512, 
                        dropout=0.1
                    )
                    #TransformerMoEBlock(#大型数据集
                    #    dim=dim,
                    #    num_experts=num_experts,
                    #    top_k=top_k,
                    #    heads=8, 
                    #    dim_head=64,
                    #    mlp_dim=512
                    #)


                )


    def forward(self, v, k, q):
        b, n, _ = q.shape
       
        hidden_list = []
        hidden_list.append(q)
        if self.pos_true:
            if self.token_len is not None:
                extra_token = repeat(self.extra_token, '1 n d -> b n d', b=b)
                q = torch.cat((extra_token, q), dim=1)
                k = torch.cat((extra_token, k), dim=1)
                v = torch.cat((extra_token, v), dim=1)
                q= q + self.pos_embedding[:, :n + self.token_len]
                k = k + self.pos_embedding[:, :n + self.token_len]
                v = v + self.pos_embedding[:, :n + self.token_len]
            else:
                q = q + self.pos_embedding[:, :n]
                k = k + self.pos_embedding[:, :n]
                v = v + self.pos_embedding[:, :n]
        
       
        q = self.dropout(q)
        k = self.dropout(k)
        v = self.dropout(v)
        for layer in self.layers:
            q = layer(v, k, q)
            if self.save_hidden:
                hidden_list.append(q)
        if self.save_hidden:
            return hidden_list
        else:
            return q

# ✅ 扩散模型增强的HybridMoEBlock
class DiffusionHybridMoEBlock(nn.Module):
    def __init__(self, *, num_frames=50, token_len=8, save_hidden, dim, depth, structure="LNLN", 
                num_experts=4, top_k=2, hidden_dim, 
                lsm_type="linear_attention", heads=8, mlp_dim=128,
                pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.1, pos_true=True,
                diffusion_steps=1000, diffusion_beta_start=0.0001, diffusion_beta_end=0.02,
                diffusion_noise_schedule="linear", use_diffusion=True):
        """
        结合扩散模型的HybridMoEBlock，用于SDHM的SCE和fusion layer
        
        Args:
            diffusion_steps: 扩散步数
            diffusion_beta_start: 初始噪声调度参数
            diffusion_beta_end: 最终噪声调度参数
            diffusion_noise_schedule: 噪声调度类型 ("linear", "cosine")
            use_diffusion: 是否启用扩散模型
        """
        super().__init__()
        
        # 基础HybridLinearMoEBlock参数
        self.token_len = token_len
        self.save_hidden = save_hidden
        self.pos_true = pos_true
        self.use_diffusion = use_diffusion
        self.num_steps = diffusion_steps
        
        
        # 位置编码
        if self.pos_true:
            if token_len is not None:
                self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
                self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
            else:
                self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
                self.extra_token = None
        else:
            self.pos_embedding = None
            self.extra_token = None

        self.dropout = nn.Dropout(emb_dropout)
        
        # 基础层
        self.layers = nn.ModuleList()
        for i in range(depth):
            block_type = structure[i % len(structure)]
            if block_type == "L":
                self.layers.append(
                    LinearMoEBlock(
                        dim=dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        hidden_dim=hidden_dim,
                        lsm_type=lsm_type
                    )
                )
            else:
                self.layers.append(
                    TransformerEncoder(
                        dim=dim, 
                        depth=1, 
                        heads=heads, 
                        dim_head=dim_head, 
                        mlp_dim=mlp_dim, 
                        dropout=dropout
                    )
                )
        
        # 扩散模型组件
        if self.use_diffusion:
            self.diffusion_steps = diffusion_steps
            self.diffusion_noise_schedule = diffusion_noise_schedule
            
            # 噪声调度
            if diffusion_noise_schedule == "linear":
                self.betas = torch.linspace(diffusion_beta_start, diffusion_beta_end, diffusion_steps)
            elif diffusion_noise_schedule == "cosine":
                self.betas = self._cosine_beta_schedule(diffusion_steps)
            else:
                raise ValueError(f"Unknown noise schedule: {diffusion_noise_schedule}")
            
            # 预计算扩散参数
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
            
            # 扩散预测网络 - 使用U-Net结构
            self.diffusion_predictor = UNetModel(
                in_channels=dim,
                out_channels=dim,
                hidden_dim=hidden_dim,
                time_dim=dim
            )
            
            # 条件嵌入（用于多模态融合）
            # 注意：condition的维度是 dim (128)，因为经过了multimodal_fusion处理
            self.condition_embedding = nn.Sequential(
                nn.Linear(dim, hidden_dim),  # 输入维度是 dim
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, dim),
                nn.LayerNorm(dim)
            )
            
            # 高级特征融合组件
            # 1. 注意力融合
            self.attention_fusion = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            
            # 2. 门控网络
            self.gate_network = nn.Sequential(
                nn.Linear(dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(0.1)
            )
            
            # 3. 残差融合
            self.residual_fusion = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
                nn.Sigmoid()
            )
            
            # 4. 最终特征融合
            self.feature_fusion_3 = nn.Sequential(
                nn.Linear(dim * 3, hidden_dim),  # 融合三种融合策略的结果
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
                nn.LayerNorm(dim)
            )
            self.feature_fusion_2 = nn.Sequential(
                nn.Linear(dim * 2, hidden_dim),  # 融合两种融合策略的结果
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
                nn.LayerNorm(dim)
            )

            self.multimodal_projection = nn.Sequential(
                        nn.Linear(256, 256),
                        nn.GELU(),
                        nn.LayerNorm(256),#
                        nn.Linear(256, 128)
                    )
            
            #self.CrossTransformer = CrossTransformerEncoder(
            #    dim=dim,
            #    heads=8,
            #    dim_head=64,
            #    mlp_dim=128,
            #    dropout=0.1,
            #    depth=1
            #)
    
    def to(self, device):
        """确保所有组件都移动到正确的设备上"""
        super().to(device)
        if hasattr(self, 'betas'):
            self.betas = self.betas.to(device)
            self.alphas_cumprod = self.alphas_cumprod.to(device)
            self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        return self
    
    def _cosine_beta_schedule(self, timesteps):
        """余弦噪声调度"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, x, t):
        """添加噪声"""
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod[t])
        
        # 扩展维度以匹配输入张量
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(-1, 1, 1)
        
        return sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise, noise
    
    def remove_noise(self, x, t, predicted_noise):
        """移除噪声"""
        alpha_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        
        # 扩展维度以匹配输入张量
        sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.view(-1, 1, 1)
        sqrt_alpha_t = sqrt_alpha_t.view(-1, 1, 1)
        
        # 预测的x_0
        predicted_x0 = (x - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
        
      
        return predicted_x0
    
    
    def diffusion_forward(self, x, num_steps, condition=None, return_loss=False):
        """扩散前向过程 - 按照DDPM正确思路实现（输入为特征向量）"""
        if not self.use_diffusion:
            return x
        
        batch_size = x.shape[0]
        device = x.device
        
        # 将扩散参数移到设备上
        if not hasattr(self, 'betas_device') or self.betas_device != device:
            self.betas = self.betas.to(device)
            self.alphas_cumprod = self.alphas_cumprod.to(device)
            self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
            self.betas_device = device
        
        # 条件嵌入
        if condition is not None:
            condition_emb = self.condition_embedding(condition)
            # 处理序列长度不匹配
            if condition_emb.shape[1] != x.shape[1]:
                condition_emb = torch.nn.functional.interpolate(
                    condition_emb.transpose(1, 2), 
                    size=x.shape[1], 
                    mode='linear'
                ).transpose(1, 2)
        else:
            condition_emb = None
        
        diffusion_loss = 0.0
        
        if return_loss:
            # ========== 训练模式 ==========
            # 保存原始输入
            original_x = x.clone()
            
            # 逐步去噪（训练时从原始输入开始）
            for i in range(num_steps):
                t = torch.full((batch_size,), num_steps - i - 1, device=device, dtype=torch.long)
                #【num_steps-i-1,num_steps-i-1,...,num_steps-i-1]64个
                
                # 1. 对原始特征添加噪声（用于训练）
                noisy_x, true_noise = self.add_noise(original_x, t)
                
                # 2. 将条件信息融合到噪声特征中
                if condition_emb is not None:
                    noisy_x_with_condition = noisy_x + 0.05 * condition_emb
                    #noisy_x_with_condition = self.CrossTransformer(noisy_x,condition_emb)
                else:
                    noisy_x_with_condition = noisy_x
                
                # 3. 预测噪声（基于噪声特征+条件）
                predicted_noise = self.diffusion_predictor(noisy_x_with_condition, t.float())
                
                # 确保预测噪声的维度与输入匹配
                if predicted_noise.shape != noisy_x.shape:
                    if predicted_noise.shape[-1] != noisy_x.shape[-1]:
                        if not hasattr(self, 'output_projection'):
                            self.output_projection = nn.Linear(predicted_noise.shape[-1], noisy_x.shape[-1]).to(device)
                        predicted_noise = self.output_projection(predicted_noise)
                    
                    if predicted_noise.shape[1] != noisy_x.shape[1]:
                        predicted_noise = torch.nn.functional.interpolate(
                            predicted_noise, 
                            size=noisy_x.shape[1], 
                            mode='linear'
                        )
                
                # 4. 计算损失：预测噪声 vs 真实噪声
                step_loss = F.mse_loss(predicted_noise, true_noise)
                diffusion_loss += step_loss
                
                # 5. 去噪（仅用于最终输出）
                if i == num_steps - 1:
                    # 使用预测噪声去噪，保持训练和推理的一致性
                    x = self.remove_noise(noisy_x, t, predicted_noise)
                    # 注意：这里使用预测噪声，因为：
                    # 1. 训练目标：让扩散模型学会预测噪声
                    # 2. 推理时：只能使用预测噪声（没有真实噪声）
                    # 3. 一致性：训练和推理使用相同的逻辑
            
            return x, diffusion_loss / num_steps
            
        else:
            # ========== 推理模式 ==========
            # 从噪声开始
            x = torch.randn_like(x)
            
            # 逐步去噪
            for i in range(num_steps):
                t = torch.full((batch_size,), num_steps - i - 1, device=device, dtype=torch.long)
                
                # 1. 将条件信息融合到当前噪声特征中
                if condition_emb is not None:
                    x_with_condition = x + 0.05 * condition_emb
                    #x_with_condition = self.CrossTransformer(x,condition_emb)
                else:
                    x_with_condition = x
                
                # 2. 预测噪声
                predicted_noise = self.diffusion_predictor(x_with_condition, t.float())
                
                # 确保预测噪声的维度与输入匹配
                if predicted_noise.shape != x.shape:
                    if predicted_noise.shape[-1] != x.shape[-1]:
                        if not hasattr(self, 'output_projection'):
                            self.output_projection = nn.Linear(predicted_noise.shape[-1], x.shape[-1]).to(device)
                        predicted_noise = self.output_projection(predicted_noise)
                    
                    if predicted_noise.shape[1] != x.shape[1]:
                        predicted_noise = torch.nn.functional.interpolate(
                            predicted_noise, 
                            size=x.shape[1], 
                            mode='linear'
                        )
                
                # 3. 去噪
                x = self.remove_noise(x, t, predicted_noise)
            
            return x
    
    def forward(self, v, k, q, condition=None, use_diffusion=True):
        """
        前向传播
        
        Args:
            v, k, q: 输入特征
            condition: 条件信息（用于多模态融合）
            use_diffusion: 是否使用扩散模型
        """
        b, n, _ = q.shape
        hidden_list = []
        hidden_list.append(q)
        
        # 位置编码
        if self.pos_true:
            if self.token_len is not None:
                extra_token = repeat(self.extra_token, '1 n d -> b n d', b=b)
                q = torch.cat((extra_token, q), dim=1)
                k = torch.cat((extra_token, k), dim=1)
                v = torch.cat((extra_token, v), dim=1)
                q = q + self.pos_embedding[:, :n + self.token_len]
                k = k + self.pos_embedding[:, :n + self.token_len]
                v = v + self.pos_embedding[:, :n + self.token_len]
            else:
                q = q + self.pos_embedding[:, :n]
                k = k + self.pos_embedding[:, :n]
                v = v + self.pos_embedding[:, :n]
        
        # Dropout
        q = self.dropout(q)
        k = self.dropout(k)
        v = self.dropout(v)
        
        # 基础层处理
        for layer in self.layers:
            q = layer(v, k, q)
            if self.save_hidden:
                hidden_list.append(q)
        
        # 扩散模型融合处理
        diffusion_loss = 0.0
        if use_diffusion and self.use_diffusion:
            if torch.allclose(v, q, atol=1e-6):  # 训练好后v约等于q，不融合以免影响效果
                print("v约等于q，不融合以免影响效果")
                multimodal_input = q
            else:  
                 # 压缩到原始维度
            # 在sce中，v: 浅层表示（原始+位置+token） q: 深层表示（多层MoE处理后） fusion: 浅层+深层的综合表示
            
            
                multimodal_input = torch.cat([v, q], dim=-1)  # [B, T, 256]
                
                # 通过线性层压缩到原始维度
                multimodal_input = self.multimodal_projection(multimodal_input)  # [B, T, 128]
            
            # 对特征进行扩散增强
            return_loss = self.training  # 只有在训练时才计算损失
            diffusion_result = self.diffusion_forward(multimodal_input, self.num_steps, condition, return_loss)
            
            if isinstance(diffusion_result, tuple):
                diffusion_enhanced, step_diffusion_loss = diffusion_result
                diffusion_loss = step_diffusion_loss
            else:
                diffusion_enhanced = diffusion_result
            
            # 根据训练/推理模式选择不同策略
            if self.training:
                # 训练时：扩散输出≈原始输入，直接使用扩散输出。不要使用融合策略，因为还原的特征和原始特征差异不大，没必要融合两个相同特征
                #################################################################
                #################################################################
                q = diffusion_enhanced
            else:
                # 推理模式：保存原始特征，然后进行融合
                # 保存原始（扩散前）特征，用于可视化对比
                base_before_fusion = q.clone().detach()
                
                # 1. 注意力融合
                attention_fused, _ = self.attention_fusion(q, diffusion_enhanced, diffusion_enhanced)
                # 2. 门控融合
                gate = torch.sigmoid(self.gate_network(torch.cat([q, diffusion_enhanced], dim=-1)))
                gated_fused = gate * q + (1 - gate) * diffusion_enhanced
                # 3. Highway-style 残差门控融合
                residual_gate = self.residual_fusion(q)
                residual_fused = diffusion_enhanced * residual_gate + q * (1 - residual_gate)
                
                # 最终融合
                q_final = self.feature_fusion_3(torch.cat([attention_fused, gated_fused, residual_fused], dim=-1))
                q = q_final
                
                # 缓存用于对比可视化的两种特征：融合前与融合后
                self._last_base_q = base_before_fusion
                self._last_fused_q = q_final.detach()
        
        if self.save_hidden:
            if diffusion_loss > 0:
                return hidden_list, q, diffusion_loss
            else:
                return hidden_list, q
        else:
            if diffusion_loss > 0:
                return q, diffusion_loss
            else:
                return q

    def get_base_and_fused(self):
        """返回最近一次推理融合前的 q 与最终融合后的 q（若不存在返回 None, None）。"""
        base_q = getattr(self, '_last_base_q', None)
        fused_q = getattr(self, '_last_fused_q', None)
        
        
        return base_q, fused_q

# ✅ 专门用于SCE的扩散增强模块
class DiffusionSCEBlock(nn.Module):
    def __init__(self,structure="NN",lsm_type='linear_attention',dim=128, depth=2, num_experts=5, top_k=2, hidden_dim=256,
                 diffusion_steps=30,use_diffusion=True):
        """
        专门用于SDHM的SCE（Context Enhancement Unit）的扩散增强模块
        """
        super().__init__()
        self.use_diffusion = use_diffusion
        
        
        self.diffusion_moe = DiffusionHybridMoEBlock(
            dim=dim,
            depth=depth,
            structure=structure,  # 交替使用Linear和Transformer
            num_experts=num_experts,
            top_k=top_k,
            hidden_dim=hidden_dim,
            lsm_type=lsm_type,
            save_hidden=True,
            token_len=None,
            pos_true=True,
            use_diffusion=use_diffusion,
            diffusion_steps=diffusion_steps
        )
        
        # 上下文增强层
        self.context_enhancer = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim)
        )
        
        # 多模态融合层
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),  # 融合文本、音频、视觉
            nn.GELU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, text_feat, audio_feat, visual_feat, return_vis_pairs=False):
        """
        前向传播
        
        Args:
            text_feat: 文本特征 [B, T, D]
            audio_feat: 音频特征 [B, T, D]
            visual_feat: 视觉特征 [B, T, D]
            return_diffusion_steps: 是否返回扩散步骤特征
        """
        
        #####################################################################
        #####################################################################
        # 多模态融合作为条件
        multimodal_condition = torch.cat([text_feat, audio_feat, visual_feat], dim=-1)
        condition = self.multimodal_fusion(multimodal_condition)
        #condition = text_feat
        
        # 扩散增强的文本处理
        diffusion_result = self.diffusion_moe(
            v=text_feat, k=text_feat, q=text_feat,
            condition=condition,
            use_diffusion=self.use_diffusion
        )
        
        # 处理返回结果
        if isinstance(diffusion_result, tuple):
            if len(diffusion_result) == 3:
                h_t_list, enhanced_text, diffusion_loss = diffusion_result
            elif len(diffusion_result) == 2:
                # 检查第一个元素是否是list（hidden_list）
                if isinstance(diffusion_result[0], list):
                    h_t_list, enhanced_text = diffusion_result
                    diffusion_loss = 0.0
                else:
                    enhanced_text, diffusion_loss = diffusion_result
                    h_t_list = [enhanced_text]  # 包装成list
            else:
                h_t_list = diffusion_result
                enhanced_text = diffusion_result
                diffusion_loss = 0.0
        else:
            h_t_list = [diffusion_result]  # 包装成list
            enhanced_text = diffusion_result
            diffusion_loss = 0.0
        
        # 上下文增强
        # 保持原始信息的同时进行轻微调整
        enhanced_text = enhanced_text + 0.05 * self.context_enhancer(enhanced_text)
        
        if return_vis_pairs and self.use_diffusion:
            base_feat, final_feat = self.diffusion_moe.get_base_and_fused()
            return h_t_list, enhanced_text, diffusion_loss, base_feat, final_feat
        
        return h_t_list, enhanced_text, diffusion_loss

