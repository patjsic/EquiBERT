'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
from itertools import cycle
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask

TQDM_DISABLE=False
writer = SummaryWriter()


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

class RRScheduler:
    '''
    Scheduling module that cycles through batches from each dataset.
    '''
    def __init__(self, dataloaders):
        '''
        Takes in dictionary of dataloaders of form {'name' : dataloader()}.

        Calling iter(dataloader) creates a permanent iter object that will return
        the same batch when reinitialized. To fix this, we create the 
        iter objects a head of time.
        '''
        self.idx = 0 #Initialize index for round robin iterations
        self.dataloaders = dataloaders
        self.names = list(dataloaders.keys())
        self.passes = {}

        #Initialize passes list
        for key in self.names:
            self.passes[key] = 0
        
        #Turn dataloaders into dataloader iterators
        self.iter_dataloaders = {}
        for key in self.names:
            self.iter_dataloaders[key] = iter(self.dataloaders[key])
    
    def end_epoch(self):
        '''
        If all values of self.passes > 0, return flag to stop epoch
        '''
        for key in self.passes.keys():
            if self.passes[key] == 0:
                return False
        return True
    
    def reset(self):
        '''
        Reset iterators.
        '''
        for key in self.names:
            self.iter_dataloaders[key] = iter(self.dataloaders[key])

    def get_batch(self):
        '''
        Get current batch depending on the index variable % 3.
        - SST
        - Para
        - STS
        '''
        dl = self.names[self.idx] #Get the key corresponding to the cycle index
        self.idx = (self.idx + 1) % len(self.names)
        # print(self.passes)
        try:
            batch = next(self.iter_dataloaders[dl])

        except StopIteration:
            print(f"RESET {dl}")
            
            #TODO: Add stop flag for largest dataset
            self.passes[dl] += 1
            print(self.passes)
            self.iter_dataloaders[dl] = cycle(self.dataloaders[dl])
            batch = next(self.iter_dataloaders[dl])

        end_epoch = self.end_epoch()

        #Reset self.passes at the end of the epoch
        if end_epoch:
            for key in self.names:
                self.passes[key] = 0
            self.reset()

            print(self.passes)

        return dl, batch, end_epoch
        
class CanonicalNetwork(nn.Module):
    '''
    BERT Module finetuned on alignnment and uniformity.
    Loss implementations taken from https://arxiv.org/pdf/2005.10242.pdf
    '''
    def __init__(self, pretrain=False):
        super(CanonicalNetwork, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            if pretrain:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def l_align(self, x, y, alpha=2):
        '''
        Alignment loss.
        '''
        # print(x.shape)
        # print(y.shape)
        return (x - y).norm(dim=1).pow(alpha).mean()

    def l_uniform(self, x, t=2):
        '''
        Uniformity loss.
        '''
        sq_pdist = torch.pdist(x, p=2).pow(2)
        sq_pdist_t = sq_pdist.mul(-t)
        # print(sq_pdist)
        return sq_pdist_t.exp().mean().log()
        
    def forward(self, input_ids, attention_mask):
        '''
        Generate two bert embeddings for contrastive learning.
        '''
        output_1 = self.bert(input_ids, attention_mask)
        output_2 = self.bert(input_ids, attention_mask)
        return output_1, output_2
    
    def calc_loss(self, x, y):
        loss = 1.5*self.l_align(x, y) + 0.25*((self.l_uniform(x) + self.l_uniform(y)) / 2)
        # print(loss)
        return loss


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config, canon_model=None, canon_path=None):
        super(MultitaskBERT, self).__init__()
        if canon_model:
            self.bert = canon_model
        elif canon_path:
            self.bert = torch.load(canon_path)
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain' or canon_model or canon_path:
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        print(self.state_dict().keys())
        if canon_model or canon_path:
            self.has_canon = True
        else:
            self.has_canon = False
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        self.sentiment = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                nn.ReLU(),
                                                nn.Linear(config.hidden_size, config.hidden_size),
                                                nn.ReLU(),
                                                nn.Linear(config.hidden_size, 5))
        self.paraphrase = nn.Sequential(nn.Linear(2*config.hidden_size, config.hidden_size),
                                                nn.ReLU(),
                                                nn.Linear(config.hidden_size, config.hidden_size),
                                                nn.ReLU(),
                                                nn.Linear(config.hidden_size, 1))
        
        self.similarity = nn.Sequential(nn.Linear(2*config.hidden_size, config.hidden_size),
                                                nn.ReLU(),
                                                nn.Linear(config.hidden_size, config.hidden_size),
                                                nn.ReLU(),
                                                nn.Linear(config.hidden_size, 1))

        #If using canonicalization network, neet to change parameters in state dict from bert.bert.layer to just bert.layer
        if self.has_canon:
            for key in list(self.state_dict().keys()):
                self.state_dict()[key.replace('bert.bert.', 'bert.')] = self.state_dict().pop(key)
        print(self.state_dict().keys())

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        if self.has_canon:
            output, _ = self.bert(input_ids, attention_mask)
        else:
            output = self.bert(input_ids, attention_mask)
        cls = output['pooler_output']
        return cls #Just output the BERT embeddings


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        if self.has_canon:
            output, _ = self.bert(input_ids, attention_mask)
        else:
            output = self.bert(input_ids, attention_mask)
        cls = output['pooler_output']
        return self.sentiment(cls)

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        #Get BERT sentence embeddings for both sentences
        if self.has_canon:
            output_1, _ = self.bert(input_ids_1, attention_mask_1)
            output_2, _ = self.bert(input_ids_2, attention_mask_2)
        else:
            output_1 = self.bert(input_ids_1, attention_mask_1)
            output_2 = self.bert(input_ids_2, attention_mask_2)
        cls_1 = output_1['pooler_output']
        cls_2 = output_2['pooler_output']

        #Concatenate both embeddings to get 2*hidden_size features
        cls_cat = torch.cat([cls_1, cls_2], dim=1)
        return self.paraphrase(cls_cat)

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        #Get BERT sentence embeddings for both sentences
        if self.has_canon:
            output_1, _ = self.bert(input_ids_1, attention_mask_1)
            output_2, _ = self.bert(input_ids_2, attention_mask_2)
        else:
            output_1 = self.bert(input_ids_1, attention_mask_1)
            output_2 = self.bert(input_ids_2, attention_mask_2)
        cls_1 = output_1['pooler_output']
        cls_2 = output_2['pooler_output']

        #Concatenate both embeddings to get 2*hidden_size features
        cls_cat = torch.cat([cls_1, cls_2], dim=1)
        return self.paraphrase(cls_cat)

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    #Set drop_last=True to avoid final batch size being the incorrect size
    #Load SST Data
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn, drop_last=True)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn, drop_last=True)

    #Load paraphrase Data
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=para_train_data.collate_fn, drop_last=True)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn, drop_last=True)
    
    #Load STS Data
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sts_train_data.collate_fn, drop_last=True)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn, drop_last=True)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)
    sim = torch.nn.CosineSimilarity()

    if args.train_canonical and args.canonical_path == None:
        canon_model = CanonicalNetwork(pretrain=False)
        canon_model = canon_model.to(device)

        lr = args.lr
        canon_optimizer = AdamW(canon_model.parameters(), lr=lr)
        best_dev_acc = 0

        #Train canonical network
        for epoch in tqdm(range(args.canonical_epochs)):
            canon_model.train()
            train_loss = 0.0
            num_batches = 0
            for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
                b_ids, b_mask, b_labels = (batch['token_ids'],
                            batch['attention_mask'], batch['labels'])

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                canon_optimizer.zero_grad()

                h, h_plus = canon_model(b_ids, b_mask)
                h = h['pooler_output']
                h_plus = h_plus['pooler_output']
                loss = canon_model.calc_loss(h, h_plus) / args.batch_size

                loss.backward()
                canon_optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss = train_loss / (num_batches)

            # train_acc, train_f1, *_  = model_eval_sst(sst_train_dataloader, canon_model, device)
            # dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, canon_model, device)

            # if dev_acc > best_dev_acc:
            #     best_dev_acc = dev_acc
            save_model(canon_model, canon_optimizer, args, config, f"canon-model-{args.canonical_epochs}-{args.lr}-multitask.pt")

            print(f"Epoch {epoch}: train loss :: {train_loss :.3f}")

        model = MultitaskBERT(config, canon_model=canon_model)
    elif args.load_canonical:
        model = MultitaskBERT(config, canon_path=args.canonical_path)
    else:
        model = MultitaskBERT(config)

    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    train_dataloaders = {"sst": sst_train_dataloader}#, "sts": sts_train_dataloader}#"para": para_train_dataloader, "sts": sts_train_dataloader}
    rr_loader = RRScheduler(train_dataloaders)
    num_batch_per_epoch = 128

    # Run for the specified number of epochs.
    for epoch in tqdm(range(args.epochs)):
        model.train()
        #iterate through each dataset
        train_loss = 0
        num_batches = 0
        end_epoch = False
        # for i in tqdm(range(num_batch_per_epoch)):
        while not end_epoch:
            key, batch, end_epoch = rr_loader.get_batch()
            if key == "sst":
                b_ids, b_mask, b_labels = (batch['token_ids'],
                                        batch['attention_mask'], batch['labels'])

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)
            else:
                b_ids1, b_mask1,b_ids2, b_mask2, b_labels, b_sent_ids = (batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'], batch['sent_ids'])
                
                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)
                b_labels = b_labels.to(device)
            # print(len(b_labels))
            optimizer.zero_grad()

            #SST uses cross entropy, but para and STS are binary
            if key == "sst":
                h = model.predict_sentiment(b_ids, b_mask)
                h_plus = model.predict_sentiment(b_ids, b_mask)
                ce_loss = F.cross_entropy(h, b_labels.view(-1), reduction='sum') / args.batch_size
            elif key == "para":
                h = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                h_plus = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                ce_loss = F.binary_cross_entropy_with_logits(h.squeeze(1), b_labels.float(), reduction='sum') / args.batch_size
            elif key == "sts":
                h = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                h_plus = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                ce_loss = F.mse_loss(h.squeeze(1), b_labels.float(), reduction='sum') / args.batch_size
            else:
                raise ValueError(f"Task label {key} not recognized.")

            #If contrastive learning, calculate contrastive loss and add to loss term
            if args.mode == "simcse":
                sim = F.cosine_similarity(h.unsqueeze(1), h_plus.unsqueeze(0), dim=-1) / args.temp
                labels = torch.arange(args.batch_size).to(device)

                # print(sim)
                # print(sim.shape)

                #Calculate simCSE loss term
                sim_loss = F.cross_entropy(sim, labels) #maximize diagonal elements
                loss = args.lambda1 * sim_loss + args.lambda2 * ce_loss
            
            #For default no contrastive learning just use cross entropy loss
            else:    
                loss = ce_loss

            loss.backward()
            optimizer.step()
            # writer.flushsts
            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        # train_sentiment_accuracy,train_sst_y_pred, train_sst_sent_ids, \
        #     train_paraphrase_accuracy, train_para_y_pred, train_para_sent_ids, \
        #     train_sts_corr, train_sts_y_pred, train_sts_sent_ids = model_eval_multitask(sst_train_dataloader,
        #                                                             para_train_dataloader,
        #                                                             sts_train_dataloader, model, device)
        
        # dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
        #     dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
        #     dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
        #                                                             para_dev_dataloader,
        #                                                             sts_dev_dataloader, model, device)

        # if dev_sentiment_accuracy > best_dev_acc:
        #     best_dev_acc = dev_sentiment_accuracy
        save_model(model, optimizer, args, config, args.filepath)
        
        # writer.add_scalar("Loss/train", train_loss, epoch)
        # writer.add_scalar("Acc/Train", sst_train_acc, epoch)
        # writer.add_scalar("Acc/Dev", sst_dev_acc, epoch)
        # writer.flush()

        print_str = f"Epoch {epoch}: train loss :: {train_loss :.3f},"
            #   f"sst train acc :: {train_sentiment_accuracy :.3f}, sst dev acc :: {dev_sentiment_accuracy :.3f},"\
            #   f"para train acc :: {train_paraphrase_accuracy :.3f}, para dev acc :: {dev_paraphrase_accuracy :.3f},"\
            #   f"sts train acc :: {train_sts_corr :.3f}, sts dev acc :: {dev_sts_corr :.3f}"
        print(print_str)


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--mode", type=str, help="default: default training loop; simcse: train using contrastive learning",
                        choices=('default', 'simcse'), default='default')
    parser.add_argument("--train_canonical", action='store_true', help="train canonical network to use as pretrained bert embedding")
    parser.add_argument("--canonical_path", type=str, 
                        help="load canonical network from given path, only if train_canonical is false", default=None)
    parser.add_argument("--canonical_epochs", type=int, default=3)
    parser.add_argument("--temp", type=float, help="temperature value for simCSE loss objective",
                        default=1.25)
    parser.add_argument("--debug_sample", action="store_true", help="if true, select subsample of 128 batches for training")
    parser.add_argument("--lambda1", type=float, help="weight for simcse loss term", default=1)
    parser.add_argument("--lambda2", type=float, help="weight for predictor loss term", default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)
