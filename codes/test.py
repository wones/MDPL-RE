import json
import random

import torch
from transformers import BertConfig, BertTokenizer,BertModel
from models.bert_for_ner import BertCrfForNer
from processors.utils_ner import get_entities
from models.layers.saam import SAAM

class Inputs:
    def __init__(self,tokenizer,entityList = ['虫害','作物','地理'],relationList = ['危害于','分布于'],quesionList = ['在文本中找出所有的实体','在文本中找出所有的实体关系']):
        self.entityList = entityList
        self.relationList = relationList
        self.quesionList = quesionList
        self.tokenizer = tokenizer

    def getTokenizer(self):
        return self.tokenizer

    def getSpecialIndexs(self,text):
        entitySeq = []
        relSeq = []
        questionSeq = []
        textSeq = []

        prompt,len_prompt = self.getPromptTextAndLen()
        input_text = '[CLS]' + prompt + '[SEP]' + '[T]' + text + '[SEP]'
        tokens = self.tokenizer.tokenize(input_text)
        t = 0
        len_tokens = len(tokens)
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
                t = count
            count += 1
        specialIndexs = {}
        specialIndexs['entityIndex'] = torch.tensor(entitySeq)
        specialIndexs['relIndex'] = torch.tensor(relSeq)
        specialIndexs['questionIndex'] = torch.tensor(questionSeq)
        specialIndexs['textIndex'] = torch.tensor(textSeq)
        specialIndexs['contextIndex'] = torch.tensor([i for i in range(t,len_tokens)])
        return specialIndexs

    #通过输入预定义的实体，实体关系和问题集合生成多维度提示文本
    def getPromptTextAndLen(self):
        prompt = ''
        count = 0
        for entity in self.entityList:
            prompt += '[E]' + entity
            count += 1 + len(entity)
        for rel in self.relationList:
            prompt += '[R]' + rel
            count += 1 + len(rel)
        for quesion in self.quesionList:
            prompt += '[Q]' + quesion
            count += 1 + len(quesion)
        return prompt,count

    #通过输入文本，得到模型输入形式
    def getTokensAndBertInputs(self,text,max_seq_length = 512,mask_padding_with_zero = True,pad_token = 0):
        prompt,len_prompt = self.getPromptTextAndLen()
        len_text = len(text)
        input_text = prompt + '[SEP]' + '[T]' + text

        # 0 代表第一句  1 代表第二句
        segment_ids = [0] * (len_prompt + 1)
        segment_ids = segment_ids + [1] * (len_text + 1)

        tokens = self.tokenizer.tokenize(input_text)
        len_tokens = len(tokens)
        special_tokens_count = 2

        if len_tokens > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
        tokens += ['[SEP]']
        segment_ids = segment_ids + [1]
        tokens = ['[CLS]'] + tokens
        segment_ids = [0] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [0] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        inputs = {}
        inputs['input_ids'] = torch.LongTensor(input_ids).view(-1,512)
        inputs['token_type_ids'] = torch.LongTensor(segment_ids).view(-1,512)
        inputs['attention_mask'] = torch.LongTensor(input_mask).view(-1,512)

        return tokens,inputs

def getSpecialFeatures(bertOutput,specialIndex):
    specialFeatures = torch.index_select(bertOutput,1,specialIndex)
    return specialFeatures





if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    # config = BertConfig.from_pretrained('outputsbert/checkpoint-7750',num_labels=23)
    # tokenizer = BertTokenizer.from_pretrained('outputsbert/checkpoint-7750/vocab.txt')
    # 1 向tokenzer添加特殊占位分隔符
    # special_tokens = {'additional_special_tokens':['[E]','[R]','[Q]','[T]']}
    # tokenizer.add_special_tokens(special_tokens)
    # 2 而resize_token_embeddings可以将bert的word embedding进行形状变换
    # model = BertCrfForNer.from_pretrained('outputsbert/checkpoint-7750',config = config)
    # model.resize_token_embeddings(len(tokenizer))
    # model.to(device)
    def get_inputsFeatures_json(filePath):
        lines = []
        with open(filePath, 'r', encoding='utf-8') as f:
            for line in f:
                lines.append(line)
        return lines,len(lines)

    devFilePath = 'datasets/DuIE/duie_dev.json/duie_dev.json'
    lines,len_lines = get_inputsFeatures_json(devFilePath)
    # print(lines[0])
    print(len_lines)
    seq_index = [i for i in range(len_lines)]
    random.shuffle(seq_index)
    print(seq_index)
    train_rate,dev_rate,test_rate = 6,2,2
    train_num = int(len_lines * (train_rate / 10))
    dev_num = int(len_lines * (dev_rate / 10))
    test_num = int(len_lines * (test_rate / 10))

    print(train_num,dev_num,test_num)

