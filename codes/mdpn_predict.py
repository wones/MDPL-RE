"""
@File    : mdpn_predict.py
@Author  : wones
@Time    : 2022/11/6 19:52
@Todo    :预测结果
"""
import json
import torch
from transformers import BertConfig, BertTokenizer
from models.bert_for_ner import BertMDPNForNer
from processors.ner_mdpn import Inputs, get_entity_relation
from tools.categoryHeatmap import make_heatMap
import torch.nn.functional as F

if __name__ == '__main__':
    pretrainPath = './outputs/mdpn_output/checkpoint-52000'
    #预测文本
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

    entityList, relList = get_entity_relation()
    tokenizer = BertTokenizer.from_pretrained(pretrainPath + '/vocab.txt')
    dataDeal = Inputs(tokenizer, entityList=entityList, relationList=relList, device=device)
    config = BertConfig.from_pretrained(pretrainPath)

    label2id, id2label = dataDeal.get_labels()

    num_label = len(label2id) + 1

    model = BertMDPNForNer.from_pretrained(pretrainPath, config=config, dsaam_in_hidden_dim=config.hidden_size,
                                              dsaam_out_hidden_dim=config.hidden_size // 2, num_label=num_label,
                                              device=device)
    model.to(device)

    specialIndexs = dataDeal.getSpecialIndexs()
    results = []
    output_predict_file = "outputs/tt_prediction.json"
    model.eval()

    sample = json.loads('{"text": "2017年3月25日，林郑月娥二儿子林约希用英文撰写支持短讯，并晒出林太与约希的温馨合照", "spo_list": [{"predicate": "母亲", "object_type": {"@value": "人物"}, "subject_type": "人物", "object": {"@value": "林郑月娥"}, "subject": "林约希"}]}')
    label_metrics = dataDeal.get_label_metrics(label2id, sample).numpy()
    x_labels = dataDeal.tokenizer.tokenize(sample['text'])
    print("正确样本标签：")
    make_heatMap(x_labels, x_labels, label_metrics)

    with torch.no_grad():
        inputs = dataDeal.getBertInputs(sample['text'])
        label_metrics = dataDeal.get_label_metrics(label2id, sample)
        all_input_ids = inputs['input_ids'].to(device)
        all_token_type_ids = inputs['token_type_ids'].to(device)
        all_attention_mask = inputs['attention_mask'] .to(device)
        all_label_mertics = label_metrics.to(device)

        span_logits = model(input_ids=all_input_ids, token_type_ids=all_token_type_ids,
                               attention_mask=all_attention_mask,
                               entityIndex=specialIndexs['entityIndex'],
                               relIndex=specialIndexs['relIndex'],
                               questionIndex=specialIndexs['questionIndex'],
                               textIndex=specialIndexs['textIndex']
                               )
        for index, label_mertics in enumerate(all_label_mertics):
            label_seq, _ = label_mertics.size()
            span_logits_item = span_logits[index][:label_seq, :label_seq, :]
            span_pre = torch.argmax(F.softmax(span_logits_item, dim=-1), dim=-1).view(-1).numpy()
            make_heatMap(x_labels,x_labels,span_pre)
    # print(results)

    # with open(output_predict_file, "w",encoding='utf-8') as writer:
    #     for record in results:
    #         for entity in record['entities']:
    #             entxt = record['text'][entity[1]:entity[2] + 1]
    #             print(entity[0] + ':' + entxt)
    #         print('-------------------')
    #         writer.write(json.dumps(record, ensure_ascii=False) + '\n')