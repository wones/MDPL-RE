"""
@File    : mdpnTest.py
@Author  : wones
@Time    : 2022/10/29 11:42
@Todo    :一句话描述todo
"""
from processors.ner_mdpn import Inputs
from models.bert_for_ner import BertMDPNForNer
from transformers import BertTokenizer,BertConfig
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = 'pre-train/bert-base-chinese'
    config = BertConfig.from_pretrained(path)
    tokenizer = BertTokenizer.from_pretrained(path + '/vocab.txt')
    inputDeal = Inputs(tokenizer=tokenizer,device = device)
    mdpn = BertMDPNForNer.from_pretrained(path,config = config,dsaam_in_hidden_dim=config.hidden_size,dsaam_out_hidden_dim=config.hidden_size//2,num_label=13,device=device)
    mdpn.resize_token_embeddings(len(tokenizer))
    mdpn.to(device)

    inputs = inputDeal.getBertInputs('黑菌虫在中国危害小麦')
    specialIndexs = inputDeal.getSpecialIndexs()
    span_logits = mdpn(input_ids = inputs['input_ids'],token_type_ids = inputs['token_type_ids'],attention_mask = inputs['attention_mask'],
                       entityIndex = specialIndexs['entityIndex'],
                       relIndex = specialIndexs['relIndex'],
                       questionIndex = specialIndexs['questionIndex'],
                       textIndex = specialIndexs['textIndex'])
    print(span_logits)

if __name__ == '__main__':
    main()
