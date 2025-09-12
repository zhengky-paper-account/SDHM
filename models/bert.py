'''
* @name: bert.py
* @description: Functions of BERT.
'''


import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

__all__ = ['BertTextEncoder']

TRANSFORMERS_MAP = {
    'bert': (BertModel, BertTokenizer),
    'roberta': (RobertaModel, RobertaTokenizer),
}


class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=False, transformers='bert', pretrained='bert-base-uncased'):
        super().__init__()
        '''
        作用：将原始文本转换为模型可以理解的数字序列
        主要功能：
        分词：将文本分解为子词（subwords）或词片（word pieces）
        转换为ID：将词汇转换为对应的数字ID
        添加特殊标记：如 [CLS]、[SEP]、[PAD] 等
        生成attention mask：标记哪些位置是真实内容，哪些是padding
        '''
        tokenizer_class = TRANSFORMERS_MAP[transformers][1]

        
        '''
        作用：接收tokenized的输入，通过transformer层处理，输出contextualized embeddings
        主要功能：
        嵌入层：将token ID转换为向量表示
        多头注意力：捕获词与词之间的关系
        前馈网络：进行特征变换
        输出表示：生成每个token的上下文相关向量
        '''
        model_class = TRANSFORMERS_MAP[transformers][0]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained)
        self.model = model_class.from_pretrained(pretrained)
        self.use_finetune = use_finetune
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states
