'''
* @name: dataset.py
* @description: Dataset loading functions. Note: The code source references MMSA (https://github.com/thuiar/MMSA/tree/master).
'''


import logging
import pickle
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader


__all__ = ['MMDataLoader']

logger = logging.getLogger('MSA')


class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        self.train_mode = args.train_mode
        self.datasetName = args.datasetName
        self.dataPath = args.dataPath
        self.missing_rate_eval_test = getattr(args, 'missing_rate_eval_test', 0.0)
        self.missing_seed = getattr(args, 'seed', 42)
        
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims
        }
        DATA_MAP[args.datasetName]()

    def __init_mosi(self):
        print(f"正在加载数据文件: {self.args.dataPath}")
        print(f"数据集名称: {self.args.datasetName}")
        print(f"训练模式: {self.args.train_mode}")
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
            'M': data[self.mode][self.args.train_mode+'_labels'].astype(np.float32),
            'missing_rate_l': np.zeros_like(data[self.mode][self.args.train_mode+'_labels']).astype(np.float32),
            'missing_rate_a': np.zeros_like(data[self.mode][self.args.train_mode+'_labels']).astype(np.float32),
            'missing_rate_v': np.zeros_like(data[self.mode][self.args.train_mode+'_labels']).astype(np.float32),
        }
        
        # 添加调试信息
        print(f"数据集标签范围: {self.labels['M'].min():.3f} - {self.labels['M'].max():.3f}")
        print(f"数据集标签均值: {self.labels['M'].mean():.3f}")
        print(f"数据集标签标准差: {self.labels['M'].std():.3f}")
        if self.args.datasetName == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.mode][self.args.train_mode+'_labels_'+m]

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not self.args.need_data_aligned:
            self.audio_lengths = data[self.mode]['audio_lengths']
            self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0

        # 生成缺失掩码
        if self.mode == 'train':
            missing_rate = [np.random.uniform(size=(len(data[self.mode][self.args.train_mode+'_labels']), 1)) for i in range(3)]
            
            for i in range(3):
                sample_idx = random.sample([i for i in range(len(missing_rate[i]))], int(len(missing_rate[i])/2))
                missing_rate[i][sample_idx] = 0

            self.labels['missing_rate_l'] = missing_rate[0]
            self.labels['missing_rate_a'] = missing_rate[1]
            self.labels['missing_rate_v'] = missing_rate[2]
        else:
            missing_rate = [self.missing_rate_eval_test * np.ones((len(data[self.mode][self.args.train_mode+'_labels']), 1)) for i in range(3)]
            self.labels['missing_rate_l'] = missing_rate[0]
            self.labels['missing_rate_a'] = missing_rate[1]
            self.labels['missing_rate_v'] = missing_rate[2]

        if self.args.need_truncated:
            self.__truncated()
            
        # 生成缺失版本的数据
        self._generate_missing_data()

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()
    
    def _generate_missing_data(self):
        """生成缺失版本的数据"""
        # 为文本生成缺失版本
        if self.args.use_bert:
            text_input_ids = self.text[:, 0, :]  # (batch_size, seq_len)
            text_input_mask = self.text[:, 1, :]  # (batch_size, seq_len)
            text_segment_ids = self.text[:, 2, :]  # (batch_size, seq_len)
        else:
            text_input_ids = self.text
            text_input_mask = np.ones_like(self.text[:, :, 0])  # 假设全为有效token
            text_segment_ids = np.zeros_like(self.text[:, :, 0])
            
        self.text_m, self.text_length, self.text_mask, self.text_missing_mask = self.generate_m(
            text_input_ids, text_input_mask, None,
            self.labels['missing_rate_l'], self.missing_seed, mode='text'
        )
        
        if self.args.use_bert:
            Input_ids_m = np.expand_dims(self.text_m, 1)
            Input_mask = np.expand_dims(self.text_mask, 1)
            Segment_ids = np.expand_dims(text_segment_ids, 1)
            self.text_m = np.concatenate((Input_ids_m, Input_mask, Segment_ids), axis=1)
        
        # 为音频生成缺失版本
        self.audio_m, self.audio_length, self.audio_mask, self.audio_missing_mask = self.generate_m(
            self.audio, None, self.audio_lengths if not self.args.need_data_aligned else None,
            self.labels['missing_rate_a'], self.missing_seed, mode='audio'
        )
        
        # 为视觉生成缺失版本
        self.vision_m, self.vision_length, self.vision_mask, self.vision_missing_mask = self.generate_m(
            self.vision, None, self.vision_lengths if not self.args.need_data_aligned else None,
            self.labels['missing_rate_v'], self.missing_seed, mode='vision'
        )
    
    def generate_m(self, modality, input_mask, input_len, missing_rate, missing_seed, mode='text'):
        """生成缺失掩码和缺失数据"""
        if mode == 'text':
            if input_mask is not None:
                input_len = np.argmin(input_mask, axis=1)
            else:
                input_len = np.full(modality.shape[0], modality.shape[1])
        elif mode == 'audio' or mode == 'vision':
            if input_len is not None:
                input_mask = np.array([np.array([1] * length + [0] * (modality.shape[1] - length)) for length in input_len])
            else:
                input_mask = np.ones_like(modality[:, :, 0]) if len(modality.shape) == 3 else np.ones_like(modality)
        
        np.random.seed(missing_seed)
        missing_mask = (np.random.uniform(size=input_mask.shape) > missing_rate.repeat(input_mask.shape[1], 1)) * input_mask
        
        assert missing_mask.shape == input_mask.shape
        
        if mode == 'text':
            # CLS SEG Token unchanged.
            for i, instance in enumerate(missing_mask):
                if len(instance) > 0:
                    instance[0] = 1  # CLS token
                    if input_len[i] > 1:
                        instance[input_len[i] - 1] = 1  # SEP token
            
            modality_m = missing_mask * modality + (100 * np.ones_like(modality)) * (input_mask - missing_mask)  # UNK token: 100.
        elif mode == 'audio' or mode == 'vision':
            if len(modality.shape) == 3:
                modality_m = missing_mask.reshape(modality.shape[0], modality.shape[1], 1) * modality
            else:
                modality_m = missing_mask * modality
        
        return modality_m, input_len, input_mask, missing_mask

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
        # 在训练模式下，每个epoch重新生成缺失掩码
        if (self.mode == 'train') and (index == 0):
            missing_rate = [np.random.uniform(size=(len(self.labels['M']), 1)) for i in range(3)]
            
            for i in range(3):
                sample_idx = random.sample([i for i in range(len(missing_rate[i]))], int(len(missing_rate[i])/2))
                missing_rate[i][sample_idx] = 0

            self.labels['missing_rate_l'] = missing_rate[0]
            self.labels['missing_rate_a'] = missing_rate[1]
            self.labels['missing_rate_v'] = missing_rate[2]

            # 重新生成缺失数据
            self._generate_missing_data()

        sample = {
            'raw_text': self.rawText[index],
            'text': torch.Tensor(self.text[index]), 
            'text_m': torch.Tensor(self.text_m[index]),
            'audio': torch.Tensor(self.audio[index]),
            'audio_m': torch.Tensor(self.audio_m[index]),
            'vision': torch.Tensor(self.vision[index]),
            'vision_m': torch.Tensor(self.vision_m[index]),
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


def MMDataEvaluationLoader(args):
    """专门用于评估的数据加载器，支持缺失率设置"""
    datasets = MMDataset(args, mode='test')

    dataLoader = DataLoader(datasets,
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=False)
    
    return dataLoader
