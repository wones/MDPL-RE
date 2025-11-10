from models.bert_for_ner import BertMDPNForNerNoMDP
from processors.ner_mdpn import Inputs,MyDuIEDataset,get_entity_relation,collate_fn
from transformers import BertTokenizer,BertConfig,AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch
import os
import logging
import random
import numpy as np
import torch.nn.functional as F
from callback.progressbar import ProgressBar
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score,recall_score,f1_score
import warnings
warnings.filterwarnings('ignore')
writer = SummaryWriter('./logs3')

def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

#保存和加载缓存数据
def load_and_cache_examples(save_cache_path,dataDeal,filePath,data_name = 'DuIE',data_type='train'):
    cached_features_file = os.path.join(save_cache_path, 'nomdp_cached_-{}_{}'.format(data_name,data_type))
    if os.path.exists(cached_features_file):
        print("Loading features from cached file {}".format(cached_features_file))
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at {}".format(save_cache_path))
        features = dataDeal.get_inputsFeaturesNoPrompt_json(filePath)
        print("Saving features into cached file {}".format(cached_features_file))
        torch.save(features, cached_features_file)
    # torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    dataset = MyDuIEDataset(features)
    return dataset


#得到损失值
def get_loss(loss_fn,span_logits,all_label_mertics):
    logis = 0
    batch = span_logits.size(0)
    for index,label_mertics in enumerate(all_label_mertics):
        label_seq,_ = label_mertics.size()
        span_logits_item = span_logits[index][:label_seq,:label_seq,:]
        hidden_dim = span_logits_item.size(-1)

        # print(span_logits_item.shape)
        # print(label_mertics.max())

        triu_ones = torch.triu(torch.ones(label_seq,label_seq)).view(-1)
        select_indexs = []
        for id,v in enumerate(triu_ones):
            if v == 1:
                select_indexs.append(id)
        
        select_indexs = torch.tensor(select_indexs).to(device)
        
        # 得到上三角形表项进行预测
        select_label_mertics = torch.index_select(label_mertics.reshape(-1), dim=0, index=select_indexs)
        select_span_logits = torch.index_select(span_logits_item.reshape(-1,hidden_dim), dim=0,index=select_indexs)

        logis += loss_fn(select_span_logits,select_label_mertics)
    return logis / batch

# 得到验证结果
def get_eval_result(span_logits,all_label_mertics):
    precision,recall,f1 = 0.0,0.0,0.0
    batch = span_logits.size(0)
    for index,label_mertics in enumerate(all_label_mertics):
        label_seq,_ = label_mertics.size()
        span_logits_item = span_logits[index][:label_seq,:label_seq,:]
        hidden_dim = span_logits_item.size(-1)

        triu_ones = torch.triu(torch.ones(label_seq, label_seq)).view(-1)
        select_indexs = []
        for id, v in enumerate(triu_ones):
            if v == 1:
                select_indexs.append(id)

        select_indexs = torch.tensor(select_indexs).to(device)
        
        # 得到上三角形表项进行预测
        select_label_mertics = torch.index_select(label_mertics.reshape(-1), dim=0, index=select_indexs)
        select_span_logits = torch.index_select(span_logits_item.reshape(-1, hidden_dim), dim=0,index=select_indexs)

        span_pre = torch.argmax(F.softmax(select_span_logits,dim = -1),dim = -1).view(-1).tolist()
        span_true = select_label_mertics.view(-1).tolist()
        precision += precision_score(span_true,span_pre,average='weighted')
        recall += recall_score(span_true,span_pre,average='weighted')
        f1 += f1_score(span_true,span_pre,average='weighted')
    return precision / batch ,recall / batch,f1 / batch



#训练模型
def train(trainLoader,bertMDPN,num_train_epochs,device,devLoader,dev_epoch_num = 1,do_dev = True,
          save_step = 500,save_path = 'outputs/mdpn_output_noprompt',
          weight_decay = 0.01,learning_rate = 2e-5,adam_epsilon = 1e-8,warmup_proportion = 0.1,gradient_accumulation_steps = 1):
    # 定义损失函数
    loss_fn = F.cross_entropy
    # 定义优化器
    t_total = len(trainLoader) // gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    bert_parameters = bertMDPN.bert.named_parameters()
    
    span_classifier_parameters = bertMDPN.span_classifier.named_parameters()

    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],"weight_decay": weight_decay, 'lr': learning_rate},
        {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': learning_rate},


        {"params": [p for n, p in span_classifier_parameters if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay, 'lr': 0.001},
        {"params": [p for n, p in span_classifier_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': 0.001},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= int(t_total * warmup_proportion),
                                                num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
#     if os.path.isfile(os.path.join(pretrainPath, "optimizer.pt")) and os.path.isfile(
#             os.path.join(pretrainPath, "scheduler.pt")):
#         # Load in optimizer and scheduler states
#         optimizer.load_state_dict(torch.load(os.path.join(pretrainPath, "optimizer.pt")))
#         scheduler.load_state_dict(torch.load(os.path.join(pretrainPath, "scheduler.pt")))

    tr_loss, logging_loss = 0.0, 0.0
    global_step = 0
    global_dev_step = 0
    bertMDPN.zero_grad()
    # 设置种子数
    seed_everything()
    pbar = ProgressBar(n_total=len(trainLoader), desc='Training', num_epochs=int(num_train_epochs))
    devbar = ProgressBar(n_total=len(devLoader), desc='Deving')
    for epoch in range(num_train_epochs):
        pbar.reset()
        devbar.reset()
        devbar.epoch_start(current_epoch=epoch)
        pbar.epoch_start(current_epoch=epoch)
        for step,ibatch in enumerate(trainLoader):
            bertMDPN.train()
            all_input_ids, all_token_type_ids, all_attention_mask, all_label_mertics = ibatch

            all_input_ids = all_input_ids.to(device)
            all_token_type_ids = all_token_type_ids.to(device)
            all_attention_mask = all_attention_mask.to(device)
            all_label_mertics = [a.to(device) for a in all_label_mertics]

            span_logits = bertMDPN(input_ids = all_input_ids, token_type_ids = all_token_type_ids,attention_mask = all_attention_mask
                                   )

            # print(span_logits.shape)
            # for i in all_label_mertics:
            #     print(i.shape)
            loss = get_loss(loss_fn,span_logits,all_label_mertics)
            loss.backward()
            pbar(step, {'loss': loss.item()})
            writer.add_scalar('Loss/step_loss',loss.item(),global_step)
            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            bertMDPN.zero_grad()
            global_step += 1

            if save_step > 0 and global_step % save_step == 0:
                output_dir = os.path.join(save_path, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    bertMDPN.module if hasattr(bertMDPN, "module") else bertMDPN
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_vocabulary(output_dir)
                print("Saving model checkpoint to {}".format(output_dir))
#                 torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
#                 torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
#                 print("Saving optimizer and scheduler states to {}".format(output_dir))

            if 'cuda' in str(device):
                torch.cuda.empty_cache()

        if do_dev and epoch % dev_epoch_num == 0:
            
            g_precision,g_recall,g_f1 = 0.0,0.0,0.0
            g_dev_step = 0
            for step, ibatch in enumerate(devLoader):
                bertMDPN.eval()
                all_input_ids, all_token_type_ids, all_attention_mask, all_label_mertics = ibatch

                all_input_ids = all_input_ids.to(device)
                all_token_type_ids = all_token_type_ids.to(device)
                all_attention_mask = all_attention_mask.to(device)
                all_label_mertics = [a.to(device) for a in all_label_mertics]

                span_logits = bertMDPN(input_ids=all_input_ids, token_type_ids=all_token_type_ids,
                                       attention_mask=all_attention_mask
                                       )

                pre,recall,f1 = get_eval_result(span_logits,all_label_mertics)
                devbar(step, {'pre': pre,'recall':recall,'f1':f1})
                g_precision += pre
                g_recall += recall
                g_f1 += f1
                g_dev_step += 1
                global_dev_step += 1
                writer.add_scalar('Precision/pre_step', pre, global_dev_step)
                writer.add_scalar('Recall/recall_step', recall, global_dev_step)
                writer.add_scalar('F1/f1_step', f1, global_dev_step)

        print('epoch:{},loss:{}'.format(epoch,tr_loss/global_step))
        print('epoch:{},dev_result:pre-{},recall-{},f1-{}'.format(epoch,g_precision / g_dev_step, g_recall / g_dev_step,g_f1 / g_dev_step))
        writer.add_scalar('Loss/loss_epoch', tr_loss / global_step,epoch)
        if do_dev and epoch % dev_epoch_num == 0:
            writer.add_scalar('Precision/pre_epoch',g_precision / g_dev_step,epoch)
            writer.add_scalar('Recall/recall_epoch', g_recall / g_dev_step, epoch)
            writer.add_scalar('F1/f1_epoch', g_f1 / g_dev_step, epoch)
    writer.close()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:',device)
    # 获取数据集，第一次加载数据集保存缓存
    pretrainPath = 'pre-train/bert-base-chinese'
    devFilePath = 'datasets/mini_DuIE/dev.json'
    trainFilePath = 'datasets/mini_DuIE/train.json'
    testFilePath = 'datasets/mini_DuIE/test.json'
    saveCashePath = 'datasets/mini_DuIE'


    # 得到预定义实体和关系
    entityList, relList = get_entity_relation()
    tokenizer = BertTokenizer.from_pretrained(pretrainPath + '/vocab.txt')
    dataDeal = Inputs(tokenizer, entityList=entityList, relationList=relList, device=device)
    
    max_text_len = dataDeal.get_max_text_len(trainFilePath)
    print("max_text_len:",max_text_len)
    # 获得数据集
    devDataset = load_and_cache_examples(saveCashePath, dataDeal, devFilePath, data_type='dev')
    devLoader = DataLoader(devDataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    trainDataset = load_and_cache_examples(saveCashePath, dataDeal, trainFilePath, data_type='train')
    trainLoader = DataLoader(trainDataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    # testDataset = load_and_cache_examples(saveCashePath, dataDeal, testFilePath, data_type='test')
    # testLoader = DataLoader(testDataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # 获得标签映射
    label2id, id2label = dataDeal.get_labels()
    num_label = len(label2id) + 1

    config = BertConfig.from_pretrained(pretrainPath)
    bertMDPN = BertMDPNForNerNoMDP.from_pretrained(pretrainPath, config=config, num_label=num_label,max_text_len = max_text_len,
                                              device=device)
    bertMDPN.resize_token_embeddings(len(tokenizer))
    bertMDPN.to(device)

    train(trainLoader,bertMDPN,30,device,devLoader)
