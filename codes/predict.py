import json
import torch
from transformers import BertConfig, BertTokenizer
from models.bert_for_ner import BertCrfForNer
from processors.utils_ner import get_entities

if __name__ == '__main__':
#预测文本
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    config = BertConfig.from_pretrained('outputsbert/checkpoint-7750',num_labels=23)
    tokenizer = BertTokenizer.from_pretrained('outputsbert/checkpoint-7750/vocab.txt')
    model = BertCrfForNer.from_pretrained('outputsbert/checkpoint-7750',config = config)
    model.to(device)

    id2label = {0: 'X', 1: 'B-CONT', 2: 'B-EDU', 3: 'B-LOC', 4: 'B-NAME', 5: 'B-ORG', 6: 'B-PRO', 7: 'B-RACE',
                8: 'B-TITLE', 9: 'I-CONT',
                10: 'I-EDU', 11: 'I-LOC', 12: 'I-NAME', 13: 'I-ORG', 14: 'I-PRO', 15: 'I-RACE', 16: 'I-TITLE',
                17: 'O', 18: 'S-NAME', 19: 'S-ORG', 20: 'S-RACE', 21: '[START]', 22: '[END]'}

    textlist = ['中国是联合国中里的五大常任理事国。']
    # text = '北京大学是中国的最高学府。'
    # inputs = tokenizer(text,return_tensors='pt')
    results = []
    output_predict_file = "outputs/tt_prediction.json"
    model.eval()

    with torch.no_grad():
        for text in textlist:
            inputs = tokenizer(text, return_tensors='pt')
            inputs = inputs.to(device)
            outputs = model(**inputs)
            logits = outputs[0]
            tags = model.crf.decode(logits,inputs['attention_mask'])
            tags = tags.squeeze(0).cpu().numpy().tolist()

            preds = tags[0][1:-1]  # [CLS]XXXX[SEP]

            label_entities = get_entities(preds, id2label, markup = 'bios')
            json_d = {}
            json_d['id'] = 1
            json_d['text'] = text
            json_d['tag_seq'] = " ".join([id2label[x] for x in preds])
            json_d['entities'] = label_entities
            results.append(json_d)
    # print(results)

    with open(output_predict_file, "w",encoding='utf-8') as writer:
        for record in results:
            for entity in record['entities']:
                entxt = record['text'][entity[1]:entity[2] + 1]
                print(entity[0] + ':' + entxt)
            print('-------------------')
            writer.write(json.dumps(record, ensure_ascii=False) + '\n')