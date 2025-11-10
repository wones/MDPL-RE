"""
@File    : ner_mdpn.py
@Author  : wones
@Time    : 2022/10/29 11:39
@Todo    :处理mdpn的输入
"""

import torch
import json
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tools.categoryHeatmap import make_heatMap

class Inputs:
    def __init__(self,tokenizer,device = 'cpu',max_seq_length = 512,entityList = ['虫害','作物','地理'],relationList = ['危害于','分布于'],quesionList = ['在文本中找出所有的实体','在文本中找出所有的实体关系']):
        self.entityList = entityList
        self.relationList = relationList
        self.quesionList = quesionList
        self.tokenizer = tokenizer
        self.special_tokens = {'additional_special_tokens': ['[E]', '[R]', '[Q]', '[T]']}
        self.tokenizer.add_special_tokens(self.special_tokens)
        self.device = device
        self.max_seq_length = max_seq_length

    def getTokenizer(self):
        return self.tokenizer

    def get_labels(self):
        label2id = {}
        id2label = {}
        alllist = self.entityList + self.relationList
        for id, label in enumerate(alllist):
            label2id[label] = id
            id2label[id] = label
        label2id['其他'] = len(alllist)
        id2label[len(alllist)] = '其他'
        return label2id, id2label

    #获取输入文本分词和分词后的长度
    def get_text_tokens_len(self,text):
        tokens = self.tokenizer.tokenize(text)
        len_tokens = len(tokens)
        return tokens,len_tokens

    # 找到实体在输入文本中的位置
    def get_position(self,tokens, spantoken):
        text = ''.join(tokens)
        spantxt = ''.join(spantoken)
        start_pos = text.find(spantxt)
        end_pos = start_pos + len(spantoken)
        return start_pos, end_pos

    # 获得输入样本的标签矩阵
    def get_label_metrics(self,label2id, sample_json):
        text = sample_json['text']
        tokens,len_tokens = self.get_text_tokens_len(text)
        otherId = label2id['其他']
        #生成标签的长度
        label_metrics = torch.full((len_tokens, len_tokens), otherId)
        spo_list = sample_json['spo_list']
        num_spo = len(spo_list)
        for i in range(num_spo):
            predicate = spo_list[i]['predicate']
            object_type = spo_list[i]['object_type']['@value']
            subject_type = spo_list[i]['subject_type']
            object = spo_list[i]['object']['@value']
            subject = spo_list[i]['subject']

            subject_start_pos, subject_end_pos = self.get_position(tokens, self.tokenizer.tokenize(subject))
            object_start_pos, object_end_pos = self.get_position(tokens, self.tokenizer.tokenize(object))

            subject_id = label2id[subject_type]
            object_id = label2id[object_type]
            relation_id = label2id[predicate]

            # 在label_metrics里标记实体关系
            label_metrics[subject_start_pos:subject_end_pos, subject_start_pos:subject_end_pos] = subject_id
            label_metrics[subject_start_pos:subject_end_pos, object_start_pos:object_end_pos] = relation_id
            label_metrics[object_start_pos:object_end_pos, object_start_pos:object_end_pos] = object_id
            label_metrics[object_start_pos:object_end_pos, subject_start_pos:subject_end_pos] = relation_id

        return label_metrics

    # 解析标签矩阵
    def parse_label_metrics(self,id2label,label_metrics):
        pass

    # 获得输入样本的掩码
    def get_mask_metrics(self,tokens):
        len_text = len(tokens)
        mask_metrics = torch.ones((len_text, len_text))
        return mask_metrics

    def getSpecialIndexs(self):
        entitySeq = []
        relSeq = []
        questionSeq = []
        textSeq = []

        prompt,len_prompt = self.getPromptTextAndLen()
        input_text = '[CLS]' + prompt + '[SEP]' + '[T]' + '[SEP]'
        tokens = self.tokenizer.tokenize(input_text)

        count = 0
        for token in tokens:
            if token == '[E]':
                entitySeq.append(count)
            elif token == '[R]':
                relSeq.append(count)
            elif token == '[Q]':
                questionSeq.append(count)
            elif token == '[T]':
                textSeq.append(count)
            count += 1
        specialIndexs = {}
        specialIndexs['entityIndex'] = torch.tensor(entitySeq).to(self.device)
        specialIndexs['relIndex'] = torch.tensor(relSeq).to(self.device)
        specialIndexs['questionIndex'] = torch.tensor(questionSeq).to(self.device)
        specialIndexs['textIndex'] = torch.tensor(textSeq).to(self.device)
        return specialIndexs

    #通过输入预定义的实体，实体关系和问题集合生成多维度提示文本
    def getPromptTextAndLen(self):
        prompt = ''
        count = 0
        for entity in self.entityList:
            prompt += '[E]' + entity
            count += 1 + len(self.tokenizer.tokenize(entity))
        for rel in self.relationList:
            prompt += '[R]' + rel
            count += 1 + len(self.tokenizer.tokenize(rel))
        for quesion in self.quesionList:
            prompt += '[Q]' + quesion
            count += 1 + len(self.tokenizer.tokenize(quesion))
        return prompt,count

    # 通过输入文本，得到不含提示文本的模型输入形式
    def getBertInputsNoPrompt(self, text, mask_padding_with_zero=True, pad_token=0):
        # 将提输入文本拼接得到plm输入文本
        input_text = text
        # 获取plm输入文本的输入特征
        tokens = self.tokenizer.tokenize(input_text)
        len_tokens = len(tokens)

        # 0 代表第一句  1 代表第二句
        segment_ids = [0] * (len_tokens)

        special_tokens_count = 2

        if len_tokens > self.max_seq_length - special_tokens_count:
            tokens = tokens[:(self.max_seq_length - special_tokens_count)]
        tokens += ['[SEP]']
        segment_ids = segment_ids + [0]
        tokens = ['[CLS]'] + tokens
        segment_ids = [0] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = self.max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [0] * padding_length

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        inputs = {}
        inputs['input_ids'] = torch.LongTensor(input_ids).view(-1, 512).to(self.device)
        inputs['token_type_ids'] = torch.LongTensor(segment_ids).view(-1, 512).to(self.device)
        inputs['attention_mask'] = torch.LongTensor(input_mask).view(-1, 512).to(self.device)

        return inputs



    #通过输入文本，得到模型输入形式
    def getBertInputs(self,text,mask_padding_with_zero = True,pad_token = 0):

        #得到多维度提示文本和提示文本长度
        prompt,len_prompt = self.getPromptTextAndLen()

        #将提示文本和输入文本拼接得到plm输入文本
        input_text = prompt + '[SEP]' + '[T]' + text

        #获取plm输入文本的输入特征
        tokens = self.tokenizer.tokenize(input_text)
        len_tokens = len(tokens)

        # 0 代表第一句  1 代表第二句
        segment_ids = [0] * (len_prompt + 1)
        segment_ids = segment_ids + [1] * (len_tokens - len_prompt - 1)
        special_tokens_count = 2

        if len_tokens > self.max_seq_length - special_tokens_count:
            tokens = tokens[:(self.max_seq_length - special_tokens_count)]
        tokens += ['[SEP]']
        segment_ids = segment_ids + [1]
        tokens = ['[CLS]'] + tokens
        segment_ids = [0] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = self.max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [0] * padding_length

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        inputs = {}
        inputs['input_ids'] = torch.LongTensor(input_ids).view(-1,512).to(self.device)
        inputs['token_type_ids'] = torch.LongTensor(segment_ids).view(-1,512).to(self.device)
        inputs['attention_mask'] = torch.LongTensor(input_mask).view(-1,512).to(self.device)

        return inputs

    # 获取json所有数据源
    def get_inputsFeatures_json(self,filePath):
        lines = []
        label2id, id2label = self.get_labels()
        with open(filePath, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip())
                # print(line['text'])
                try:
                    inputs = self.getBertInputs(line['text'])
                    label_metrics = self.get_label_metrics(label2id, line)
                    lines.append([inputs,label_metrics])
                except Exception as e:
                    # print(e)
                    continue
        return lines
    
    def get_max_text_len(self, filePath):
        max_len = 0
        with open(filePath, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip())
                # print(line['text'])
                try:
                    txt_len = len(self.tokenizer.tokenize(line['text']))
                    max_len = max(max_len,txt_len)
                except Exception as e:
                    # print(e)
                    continue
        return max_len

    def get_inputsFeaturesNoPrompt_json(self,filePath):
        lines = []
        label2id, id2label = self.get_labels()
        with open(filePath, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip())
                # print(line['text'])
                try:
                    inputs = self.getBertInputsNoPrompt(line['text'])
                    label_metrics = self.get_label_metrics(label2id, line)
                    lines.append([inputs,label_metrics])
                except Exception as e:
                    # print(e)
                    continue
        return lines


class MyDuIEDataset(Dataset):
    def __init__(self,datas):
        self.data = datas

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        idata = self.data[item]
        return idata

def get_entity_relation(filePath = 'datasets/DuIE/duie_schema/duie_schema.json'):
    entityList = []
    relationList = []
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            object_type = line['object_type']['@value']
            subject_type = line['subject_type']
            predicate = line['predicate']
            if object_type not in entityList:
                entityList.append(object_type)
            if subject_type not in entityList:
                entityList.append(subject_type)
            if predicate not in relationList:
                relationList.append(predicate)
    return entityList,relationList


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """

    all_label_mertics = []
    all_input_ids = torch.cat([ibatch[0]['input_ids'] for ibatch in batch],dim=0)
    all_token_type_ids = torch.cat([ibatch[0]['token_type_ids'] for ibatch in batch], dim=0)
    all_attention_mask = torch.cat([ibatch[0]['attention_mask'] for ibatch in batch], dim=0)
    for ibatch in batch:
        all_label_mertics.append(ibatch[1])

    return all_input_ids, all_token_type_ids, all_attention_mask, all_label_mertics



if __name__ == '__main__':
    def get_label_metrics(self,label2id, sample_json):
        text = sample_json['text']
        tokens,len_tokens = self.get_text_tokens_len(text)
        otherId = label2id['其他']
        #生成标签的长度
        label_metrics = torch.full((len_tokens, len_tokens), otherId)
        spo_list = sample_json['spo_list']
        num_spo = len(spo_list)
        for i in range(num_spo):
            predicate = spo_list[i]['predicate']
            object_type = spo_list[i]['object_type']['@value']
            subject_type = spo_list[i]['subject_type']
            object = spo_list[i]['object']['@value']
            subject = spo_list[i]['subject']

            subject_start_pos, subject_end_pos = self.get_position(tokens, self.tokenizer.tokenize(subject))
            object_start_pos, object_end_pos = self.get_position(tokens, self.tokenizer.tokenize(object))

            subject_id = label2id[subject_type]
            object_id = label2id[object_type]
            relation_id = label2id[predicate]

            # 在label_metrics里标记实体关系
            label_metrics[subject_start_pos:subject_end_pos, subject_start_pos:subject_end_pos] = subject_id
            label_metrics[subject_start_pos:subject_end_pos, object_start_pos:object_end_pos] = relation_id
            label_metrics[object_start_pos:object_end_pos, object_start_pos:object_end_pos] = object_id
            label_metrics[object_start_pos:object_end_pos, subject_start_pos:subject_end_pos] = relation_id

        return label_metrics
    #显示所有torch.tensor数据
    torch.set_printoptions(profile="full")

    entityList,relationList = get_entity_relation(filePath = '../datasets/DuIE/duie_schema/duie_schema.json')
    tokenizer = BertTokenizer.from_pretrained('../pre-train/bert-base-chinese/vocab.txt')
    inputDeal = Inputs(tokenizer,entityList=entityList,relationList=relationList)

    label2id,id2label = inputDeal.get_labels()
    # lines = inputDeal.get_inputsFeatures_json('../datasets/DuIE/duie_dev.json/dev.json')
    # print(lines)

    # line = json.loads('{"text": "杰瑞的冷静太空是北京联合出版公司出版的一本图书，作者是简·尼尔森博士（Dr", "spo_list": [{"predicate": "作者", "object_type": {"@value": "人物"}, "subject_type": "图书作品", "object": {"@value": "尼尔森"}, "subject": "杰瑞的冷静太空"}, {"predicate": "作者", "object_type": {"@value": "人物"}, "subject_type": "图书作品", "object": {"@value": "简"}, "subject": "杰瑞的冷静太空"}]}')
    # label_metics = inputDeal.get_label_metrics(label2id,line)
    inputs = inputDeal.getBertInputs('我爱中国-china,这是我的祖国。')
    # print(label_metics.shape)
    # print(inputs['input_ids'])
    # print(inputDeal.getTokenizer().decode(inputs['input_ids'].view(-1)))
    # print(inputs['token_type_ids'])
    # print(inputs['attention_mask'])
    # print(inputs['input_ids'].shape)
    # print(inputs['token_type_ids'].shape)
    # print(inputs['attention_mask'].shape)
    sample = json.loads('{"text": "《仙界绿化师》是无花酒创作的网络小说，发表于起点网", "spo_list": [{"predicate": "作者", "object_type": {"@value": "人物"}, "subject_type": "图书作品", "object": {"@value": "无花酒"}, "subject": "仙界绿化师"}]}')
    label_metrics = inputDeal.get_label_metrics(label2id,sample).numpy()
    x_labels = inputDeal.tokenizer.tokenize(sample['text'])
    make_heatMap(x_labels,x_labels,label_metrics)

