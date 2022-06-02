import torch
from transformers import BertTokenizer
import json
from torch.utils.data import Dataset
import os

#0.获取MUL数据
def getMulContext(filePath):
    mulData= []
    with open(filePath,encoding='utf-8') as f:
        fileData = json.load(f)
        for data in fileData:
            mulData.append(data)
    return mulData

#1.获取MUL样本数据特征表示
def getInputsFeature(mulDataIter,max_length,tokenizer):
    dataIter = mulDataIter
    # labelMap = {'NS': 1, 'NR': 2, 'NT': 3}

    context = dataIter['context']
    start_position = dataIter['start_position']
    end_position = dataIter['end_position']
    span_position = dataIter['span_position']
    span_entity_label = dataIter['span_entity_label']

    # 对中文数据集进行预处理
    context = "".join(context.split())
    new_end_position = [x + 1 for x in end_position]
    new_start_position = [x + 1 for x in start_position]
    # new_start_position = start_position
    # new_end_position = end_position

    inputs = tokenizer(context, return_tensors='pt')
    len_input_ids = inputs['input_ids'].size(-1)
    len_rest = max_length - inputs['input_ids'].size(-1)
    # print("start_position",start_position)
    # print("new_start_position",new_start_position)
    start_labels = [1 if idx in new_start_position else 0 for idx in range(len_input_ids)]
    # print("start_labels",start_labels)
    end_labels = [1 if idx in new_end_position else 0 for idx in range(len_input_ids)]

    start_labels_mask = [1] * len_input_ids
    end_labels_mask = [1] * len_input_ids

    if len_rest < 0:
        inputs['input_ids'].resize_([1, max_length])
        inputs['token_type_ids'].resize_([1, max_length])
        inputs['attention_mask'].resize_([1, max_length])
        start_labels = start_labels[:max_length]
        end_labels = end_labels[:max_length]
        start_labels_mask = start_labels_mask[:max_length]
        end_labels_mask = end_labels_mask[:max_length]

        # 确保最后一个token是[SEP]
        sep_token = tokenizer.sep_token_id
        if inputs['input_ids'][0][-1] != sep_token:
            assert inputs['input_ids'].size(-1) == max_length
            inputs['input_ids'][0][-1] = sep_token
            start_labels[-1] = 0
            end_labels[-1] = 0
            start_labels_mask[-1] = 0
            end_labels_mask[-1] = 0

    elif len_rest > 0:
        inputs['input_ids'] = torch.cat((inputs['input_ids'], torch.tensor([[100] * len_rest])), 1)
        inputs['token_type_ids'] = torch.cat((inputs['token_type_ids'], torch.tensor([[0] * len_rest])), 1)
        inputs['attention_mask'] = torch.cat((inputs['attention_mask'], torch.tensor([[0] * len_rest])), 1)
        start_labels = start_labels + [0] * len_rest
        end_labels = end_labels + [0] * len_rest
        start_labels_mask = start_labels_mask + [0] * len_rest
        end_labels_mask = end_labels_mask + [0] * len_rest

    len_seq = max_length
    match_labels = torch.zeros([len_seq, len_seq], dtype=torch.long)
    for span, label in zip(span_position, span_entity_label):
        start, end = map(int, span.split(';'))
        start = start + 1
        end = end + 1
        if start >= len_seq or end >= len_seq:
            continue
        match_labels[start, end] = label

    # 类型转换
    start_labels = torch.LongTensor(start_labels)
    end_labels = torch.LongTensor(end_labels)
    start_labels_mask = torch.LongTensor(start_labels_mask)
    end_labels_mask = torch.LongTensor(end_labels_mask)


    return [inputs,start_labels,end_labels,start_labels_mask,end_labels_mask,match_labels]


class MULDataset(Dataset):
    def __init__(self,mulData,max_length,tokenizer):
        self.mulData = mulData
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.mulData)

    def __getitem__(self, item):
        return getInputsFeature(self.mulData[item],max_length=self.max_length,tokenizer=self.tokenizer)

def getMulDataset(dataPath,dataType,max_length,tokenizer):
    filePath = dataPath + '/mul-ner.' + dataType
    muldata = getMulContext(filePath)
    mulDataset = MULDataset(muldata,max_length,tokenizer)
    return mulDataset

#2.获取样本最大不同个数
def getMaxSeqLength(dataPath,dataType):
    filePath = dataPath + '/mul-ner.' + dataType
    muldata = getMulContext(filePath)
    tokens = []
    for item in muldata:
        context = item['context']
        for token in context.split():
            if token not in tokens:
                tokens.append(token)
    return len(tokens)

if __name__=='__main__':
    tokenizer = BertTokenizer.from_pretrained('../pre-train/bert-base-chinese/vocab.txt')
    mulDataset = getMulDataset('./mul_MSRA','train',512,tokenizer)
    print(mulDataset[1][-1])
    print(getMaxSeqLength('./mul_MSRA','train'))