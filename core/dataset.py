'''
* @name: dataset.py
* @description: Dataset loading functions. Note: The code source references MMSA (https://github.com/thuiar/MMSA/tree/master).
'''


import logging
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


__all__ = ['MMDataLoader']

logger = logging.getLogger('MSA')


class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims
        }
        DATA_MAP[args.datasetName]()

    def __init_mosi(self):
        print(f"🔍 正在加载数据文件: {self.args.dataPath}")
        print(f"🔍 数据集名称: {self.args.datasetName}")
        print(f"🔍 训练模式: {self.args.train_mode}")
        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f) 
        #data['train']['text_bert'] (1368,3,39)
        #data['train']['text'] (1368,39,768)
        #data['train']['vision'] (1368,55,709)
        #data['train']['audio'] (1368,400,33)
        #data['train']['audio_text'] (1368,8,768)
        #data['train']['vision_text'] (1368,8,768)
        #data['train']['raw_text']

        self.args.use_bert = True
        self.args.need_truncated = True
        self.args.need_data_aligned = True

        if self.args.use_bert:
            self.text = data[self.mode]['text_bert'].astype(np.float32)
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
     
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.audio_text = data[self.mode]['audio_text'].astype(np.float32)
        #self.vision_text = data[self.mode]['vision_text'].astype(np.float32)
        # 视觉描述有8个，只加载前4个描述，减少计算量
        self.vision_text = data[self.mode]['vision_text'][:, :4, :].astype(np.float32)

        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']
        self.labels = {
            'M': data[self.mode][self.args.train_mode+'_labels'].astype(np.float32)#定义lables【M】
        }
        
        # 添加调试信息
        print(f"🔍 数据集标签范围: {self.labels['M'].min():.3f} - {self.labels['M'].max():.3f}")
        print(f"🔍 数据集标签均值: {self.labels['M'].mean():.3f}")
        print(f"🔍 数据集标签标准差: {self.labels['M'].std():.3f}")
        if self.args.datasetName == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.mode][self.args.train_mode+'_labels_'+m]#定义lables【T】，【A】，【V】

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not self.args.need_data_aligned:
            self.audio_lengths = data[self.mode]['audio_lengths']
            self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0

        if self.args.need_truncated:
            self.__truncated()

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def __truncated(self):
        # NOTE: Here for dataset we manually cut the input into specific length.
        def Truncated(modal_features, length):#modal_features: 模态特征数据，形状为 (batch_size, seq_len, feature_dim)
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])#创建填充向量，长度等于特征维度
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+length])#如果超出边界，说明已经接近序列末尾，无法找到更好的截断位置，则从当前位置开始截断 length 长度的数据
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+length])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
                       
        text_length, audio_length, video_length = self.args.seq_lens

        audio_length, video_length =[50,50] 
        
        self.vision = Truncated(self.vision, video_length)
        # self.text = Truncated(self.text, text_length)
        self.audio = Truncated(self.audio, audio_length)

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if self.args.use_bert:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'raw_text': self.rawText[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'vision_text':torch.Tensor(self.vision_text[index]),
            'audio_text':torch.Tensor(self.audio_text[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 
        if not self.args.need_data_aligned:
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
        return sample


   



def MMDataLoader(args):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len() 

    dataLoader = {
        'train': DataLoader(datasets['train'],
                           batch_size=args.batch_size,
                           num_workers=args.num_workers,
                           shuffle=True),  # 只有训练集shuffle
        'valid': DataLoader(datasets['valid'],
                           batch_size=args.batch_size,
                           num_workers=args.num_workers,
                           shuffle=False),  # 验证集不shuffle
        'test': DataLoader(datasets['test'],
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          shuffle=False)   # 测试集不shuffle
    }
    
    return dataLoader