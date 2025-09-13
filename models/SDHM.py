'''
* @name: SDHM.py
'''

import torch
from torch import nn
from .AMA import Transformer, CrossTransformer, AMAEncoder
from .bert import BertTextEncoder
from einops import repeat
import torch.nn.functional as F
from .hybrid_encoder import HybridLinearMoEBlock

class SDHM(nn.Module):
    def __init__(self, dataset, AMA_depth=3, fusion_layer_depth=2, bert_pretrained="D:\\tools\\bert-base-chinese", 
                 sce_diffusion=True, diffusion_beta_start=0.0001, diffusion_beta_end=0.02, 
                 diffusion_noise_schedule='linear'):
        super(SDHM, self).__init__()

        # 添加这两行
        self.log_sigma_main = nn.Parameter(torch.tensor(0.0))   # log(σ1)
        self.log_sigma_diff = nn.Parameter(torch.tensor(0.0))   # log(σ2)

        # Note that when modifying T, it needs to be modified here.
        #如果特征增强阶段使用了位置编码，h_fusion_a的第二维会变成16（8token+8pos），AMA注意力计算后h_minor_shift第二维是16，因此h_minor的shape需要[64, 16, 128]，否则[64, 8, 128]
        self.h_minor = nn.Parameter(torch.ones(1, 8, 128))#nn.Parameter：将普通张量转换为可学习参数。Parameter：会被自动注册到模块的参数列表中，参与梯度计算和优化
        
        # 扩散模型参数
        self.sce_diffusion = sce_diffusion
        #elf.diffusion_steps = diffusion_steps
        self.diffusion_beta_start = diffusion_beta_start
        self.diffusion_beta_end = diffusion_beta_end
        self.diffusion_noise_schedule = diffusion_noise_schedule

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=bert_pretrained)#可修改

        # mosi
        if dataset == 'mosi':
            self.proj_l0 = nn.Linear(768, 128)
            self.proj_a0 = nn.Linear(5, 128)
            self.proj_v0 = nn.Linear(20, 128)
            self.proj_al0 = nn.Linear(768, 128)
            self.proj_vl0 = nn.Linear(768, 128)
            # self.proj_vl0 = nn.Linear(16, 128) #onehot
        elif dataset == 'mosei':
            self.proj_l0 = nn.Linear(768, 128)
            self.proj_a0 = nn.Linear(74, 128)
            self.proj_v0 = nn.Linear(35, 128)
            self.proj_al0 = nn.Linear(768, 128)
            self.proj_vl0 = nn.Linear(768, 128)
        elif dataset == 'sims':
            self.proj_l0 = nn.Linear(768, 128)
            self.proj_a0 = nn.Linear(33, 128)
            self.proj_v0 = nn.Linear(709, 128)
            self.proj_al0 = nn.Linear(768, 128)
            self.proj_vl0 = nn.Linear(768, 128)
        else:
            assert False, "DatasetName must be mosi, mosei or sims."

        # Note that when modifying T, the Transformer needs to modify token_len
        self.proj_l = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)
        self.proj_a = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)
        self.proj_v = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)
        self.proj_al = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)
        self.proj_vl = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)

        # Note that when modifying T, text_encoder to modify num_frames, CrossTransformer to modify source_num_frames and tgt_num_frames,
        self.text_encoder = Transformer(num_frames=50, save_hidden=True, token_len=None, dim=128, depth=MFU_depth-1, heads=8, mlp_dim=128)
        self.AMA_layer = AMAEncoder(dim=128, depth=MFU_depth, heads=8, dim_head=16, dropout = 0.)
        #如果特征增强阶段使用了位置编码，两个num_frames都改为16
        self.fusion_layer = CrossTransformer(source_num_frames=8, tgt_num_frames=8, dim=128, depth=fusion_layer_depth, heads=8, mlp_dim=128)
        self.fusion_layer_concat = nn.Linear(128+128, 128) # The final fusion module is concat.
        self.fusion_layer_add = nn.Linear(128, 128) # Finally, the fusion module is added.

        # the post_vvtfusion layers
        self.vvtfusion_dropout = nn.Dropout(p=0.1)
        self.vvtfusion_layer = nn.Linear(128+128, 128) # 'video_out': 32 + 'video_text_out': 768, 32

        # the post_aatfusion layers
        self.aatfusion_dropout = nn.Dropout(p=0.1)
        self.aatfusion_layer = nn.Linear(128+128, 128) # 'audio_out': 16 + 768, 16

        # the post_avttfusion layers
        self.avttfusion_dropout = nn.Dropout(p=0.0)
        self.avttfusion_layer = nn.Linear(128+128+128, 128) # 768+768+768, 768


        
        self.vvt_moe = HybridLinearMoEBlock(
            dim=256,  # 128+128
            num_experts=5,
            top_k=2,
            hidden_dim=256,
            lsm_type='linear_attention',
            pos_true=False,
            structure="L",
            depth=1,
            save_hidden=False,
        )

        self.aat_moe = HybridLinearMoEBlock(
            dim=256,  # 128+128
            num_experts=5,
            top_k=2,
            hidden_dim=256,
            lsm_type='linear_attention',
            pos_true=False,
            structure="L",
            depth=1,
            save_hidden=False,
        )

        self.avtt_moe = HybridLinearMoEBlock(
            dim=384,  # 128+128+128
            num_experts=5,
            top_k=2,
            hidden_dim=256,
            lsm_type='linear_attention',
            pos_true=False,
            structure="L",
            depth=1,
            save_hidden=False,
        )
        self.vvt_proj = nn.Linear(256, 128)
        self.aat_proj = nn.Linear(256, 128)
        self.avtt_proj = nn.Linear(384, 128)
        

       
        # 根据参数决定是否创建扩散增强的sce
        if sce_diffusion:
            from .hybrid_encoder import DiffusionSCEBlock
            self.diffusion_sce = DiffusionSCEBlock(
                dim=128,
                depth=MFU_depth-1,
                num_experts=5,
                top_k=2,
                hidden_dim=128,
                use_diffusion=sce_diffusion,
                lsm_type='linear_attention',
                diffusion_steps=100,
                structure="NN",
            )
        
        # 保留text_encoder_moe作为备用
        self.text_encoder_moe = HybridLinearMoEBlock(
            dim=128,
            depth=AMA_depth-1,
            structure="NN",
            num_experts=5,
            top_k=2,
            hidden_dim=256,
            lsm_type='linear_attention',
            save_hidden=True,
            token_len=None,
            pos_true=True,
        )

       
        self.fusion_layer_moe = HybridLinearMoEBlock(
            dim=128,
            num_frames=8,
            save_hidden=False,
            depth=4,   
            structure="LNLN",
            num_experts=5,
            top_k=2,
            lsm_type='linear_attention',#要以文本为核心模态，只能用linear_attention
            hidden_dim=256,#expert的hidden_dim
        )
        

        self.head_moe = HybridLinearMoEBlock(
            dim=128,
            num_frames=1,
            save_hidden=False,
            depth=2,
            structure="LN",  # 交替的线性层和归一化层
            num_experts=5,
            token_len=1,
            top_k=2,
            pos_true=False,   # 启用位置编码，增强空间感知
            lsm_type='mamba',
            hidden_dim=256,
        )


        self.cls_head = nn.Sequential(
            nn.Linear(128, 1)
        )


    def forward(self, x_visual, x_audio, x_text, audio_text, vision_text, current_epoch=0, return_vis_pairs=False):
        b = x_visual.size(0)
        h_minor = repeat(self.h_minor, '1 n d -> b n d', b = b)#[b,8或16,128]的可学习参数，初始为全1.
        
        x_text = self.bertmodel(x_text)

        x_visual = self.proj_v0(x_visual)
        x_audio = self.proj_a0(x_audio)
        x_text = self.proj_l0(x_text)
        x_audio_text = self.proj_al0(audio_text)
        x_vision_text = self.proj_vl0(vision_text)

        h_v = self.proj_v(x_visual)[:, :8]
        h_a = self.proj_a(x_audio)[:, :8]
        h_t = self.proj_l(x_text)[:, :8]
        h_at = self.proj_al(x_audio_text)[:, :8]
        h_vt = self.proj_vl(x_vision_text)[:, :8]

        #下面这些特征将用于融合
        h_fusion_a = h_a
        h_fusion_v = h_v
        h_fusion_t = h_t

        # Combine audio and audio_text
       
        h_fusion_a = torch.cat([h_fusion_a, h_at], dim=-1)
        
        h_fusion_a = self.aatfusion_dropout(h_fusion_a)
 
        h_fusion_a = self.aat_moe(h_fusion_a, h_fusion_a, h_fusion_a)  # LinearMoEBlock
        h_fusion_a = self.aat_proj(h_fusion_a)
    

        # Combining vision and vision_text
        h_fusion_v = torch.cat([h_fusion_v, h_vt], dim=-1)
        h_fusion_v = self.vvtfusion_dropout(h_fusion_v)
     
       
        h_fusion_v = self.vvt_moe(h_fusion_v, h_fusion_v, h_fusion_v)  # LinearMoEBlock
        h_fusion_v = self.vvt_proj(h_fusion_v)
   


        # Combine text and audio_text vision_text
        h_fusion_t = torch.cat([h_vt, h_at, h_t], dim=-1)
        h_fusion_t = self.avttfusion_dropout(h_fusion_t)
        
        h_fusion_t = self.avtt_moe(h_fusion_t, h_fusion_t, h_fusion_t)  # LinearMoEBlock
        h_fusion_t = self.avtt_proj(h_fusion_t)
  

       
        # 使用扩散增强的sce处理
        sce_diffusion_loss = 0.0
        if self.sce_diffusion and hasattr(self, 'diffusion_sce'):
            #print("使用扩散增强的sce")
            # 使用扩散增强的sce
            sce_result = self.diffusion_sce(h_fusion_t, h_fusion_a, h_fusion_v, return_vis_pairs=return_vis_pairs)
            
            if isinstance(sce_result, tuple):
                if len(sce_result) >= 5:  # 包含可视化特征
                    h_t_list, enhanced_text, sce_diffusion_loss, base_vis, final_vis = sce_result
                    self.saved_base_vis = base_vis
                    self.saved_final_vis = final_vis
                else:
                    h_t_list, enhanced_text, sce_diffusion_loss = sce_result
                    self.saved_base_vis = None
                    self.saved_final_vis = None
            else:
                h_t_list, enhanced_text = sce_result
                sce_diffusion_loss = 0.0
                self.saved_base_vis = None
                self.saved_final_vis = None
        else:
            # 使用原始的sce
            h_t_list = self.text_encoder_moe(h_fusion_t, h_fusion_t, h_fusion_t) # h_t_list = self.text_encoder(h_fusion_t)
            enhanced_text = h_t_list[-1] if isinstance(h_t_list, list) else h_t_list
            self.saved_base_vis = None
            self.saved_final_vis = None
        
        h_minor = self.AMA_layer(h_t_list, h_fusion_a, h_fusion_v, h_minor) # ([64, 8, 128])
        
        
           




        #前期不使用sce扩散增强的文本特征，后期把sce扩散增强文本和hiddenlist[-1]融合送入fusion_layer_moe
        if current_epoch >= 0:#mosi100，sims77
            print("融合扩散生成结果")
            #feat = self.fusion_layer_moe(h_minor, h_minor, enhanced_text)[:, 0] # ([64, 128])#新代码3
            alpha = 0.4
            feat = (1 - alpha) * h_t_list[-1] + alpha * enhanced_text
            feat = self.fusion_layer_moe(h_minor, h_minor, feat)[:, 0] 
        else:
            feat = self.fusion_layer_moe(h_minor, h_minor, h_t_list[-1])[:, 0] # ([64, 128])#新代码3
             
    
        feat = feat.unsqueeze(1)
        feat = self.head_moe(feat, feat, feat)[:, 0]
        output = self.cls_head(feat)

        # 为可视化返回融合前后两种特征
        base_fusion_feat = h_t_list[-1][:, 0] if isinstance(h_t_list, list) else h_t_list[:, 0]
        diff_fusion_feat = feat

        total_diffusion_loss = sce_diffusion_loss 

        if total_diffusion_loss > 0:
            return output, total_diffusion_loss
        else:
            return output
    
    


def build_model(opt):
    if opt.datasetName == 'sims':
        l_pretrained='/share/home/zhengky/bert-base-chinese'
    else:
        l_pretrained='/share/home/zhengky/bert-base-uncased'
    print(l_pretrained)
    
    # 获取扩散模型参数
 
    sce_diffusion = getattr(opt, 'sce_diffusion') 
    diffusion_beta_start = getattr(opt, 'diffusion_beta_start', 0.0001)
    diffusion_beta_end = getattr(opt, 'diffusion_beta_end', 0.02)
    diffusion_noise_schedule = getattr(opt, 'diffusion_noise_schedule', 'linear')
    
    print(f"Diffusion settings:")

    print(f"  SCE diffusion: {sce_diffusion}")
    print(f"  Beta start: {diffusion_beta_start}")
    print(f"  Beta end: {diffusion_beta_end}")
    print(f"  Noise schedule: {diffusion_noise_schedule}")
    
    model = SDHM(
        dataset=opt.datasetName, 
        fusion_layer_depth=opt.fusion_layer_depth, 
        bert_pretrained=l_pretrained,
        sce_diffusion=sce_diffusion,
        diffusion_beta_start=diffusion_beta_start,
        diffusion_beta_end=diffusion_beta_end,
        diffusion_noise_schedule=diffusion_noise_schedule
    )

    return model




