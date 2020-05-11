from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from transformers import BertForSequenceClassification
from copy import deepcopy
import gc
import torch
from sklearn.metrics import accuracy_score
import numpy as np


class Learner(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args):
        """
        :param args:
        """
        super(Learner, self).__init__()

        self.num_labels = args.num_labels
        self.outer_batch_size = args.outer_batch_size
        self.inner_batch_size = 1
        self.outer_update_lr = args.outer_update_lr
        self.inner_update_lr = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.inner_update_step_eval = args.inner_update_step_eval
        self.bert_model = args.bert_model
        self.meta_testing_size = args.meta_testing_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = BertForSequenceClassification.from_pretrained(self.bert_model, num_labels=self.num_labels)
        self.outer_optimizer = Adam(self.model.parameters(), lr=self.outer_update_lr)
        self.model.train()

    def forward(self,ids, batch_tasks):

        task_accs = []
        num_task = len(batch_tasks)

        for task_id, task in enumerate(batch_tasks):
            support = task[0]
            query = task[1]
            # Random Initialize W_ for classification
            torch.nn.init.xavier_uniform_(self.model.classifier.weight.data)

            self.model.to(self.device)
            support_dataloader = DataLoader(support, sampler=RandomSampler(support),
                                            batch_size=self.inner_batch_size)

            inner_optimizer = Adam(self.model.parameters(), lr=self.inner_update_lr)

            self.model.train()
            for name, param in self.model.named_parameters():
                if 'classifier' not in name:  # classifier layer
                    param.learn = False
                else:
                    param.learn = True

            print('----Task', ids[task_id], '----')
            all_loss = []
            for inner_step, batch in enumerate(support_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, segment_ids, label_id = batch
                outputs = self.model(input_ids, attention_mask, segment_ids, labels=label_id)
                loss = outputs[0]
                loss.backward()
                inner_optimizer.step()
                inner_optimizer.zero_grad()
                all_loss.append(loss.item())
            print("Loss in support set: ", np.mean(all_loss))

            print('----Training Outer Step-----')
            query_dataloader = DataLoader(query, sampler=None, batch_size=len(query))
            query_batch = iter(query_dataloader).next()
            query_batch = tuple(t.to(self.device) for t in query_batch)
            q_input_ids, q_attention_mask, q_segment_ids, q_label_id = query_batch

            q_outputs = self.model(q_input_ids, q_attention_mask, q_segment_ids, labels=q_label_id)

            for name, param in self.model.named_parameters():
                if 'classifier' in name:  # classifier layer
                    param.learn = False
                else:
                    param.learn = True

            q_loss = q_outputs[0]
            q_loss.backward()
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()

            q_logits = F.softmax(q_outputs[1], dim=1)
            pre_label_id = torch.argmax(q_logits, dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = q_label_id.detach().cpu().numpy().tolist()

            acc = accuracy_score(pre_label_id, q_label_id)
            task_accs.append(acc)
            print('Acc in query set: ', acc)

            del inner_optimizer
            torch.cuda.empty_cache()
            gc.collect()
            self.model.to(torch.device('cpu'))

        return np.mean(task_accs)

    def finetune(self, idt, batch_tasks):
        task_accs = []
        num_task = len(batch_tasks)
        for task_id, task in enumerate(batch_tasks):
            support = task[0]
            query = task[1]
            self.model.to(self.device)
            self.model.train()
            # Random Initialize W_ for speciffic task
            torch.nn.init.xavier_uniform_(self.model.classifier.weight.data)
            
            support_dataloader = DataLoader(support, sampler=RandomSampler(support, replacement=True, num_samples = self.meta_testing_size),
                                            batch_size=self.inner_batch_size)
            inner_optimizer = Adam(self.model.parameters(), lr=self.inner_update_lr)

            # NOTE: Below is based on the MRCL paper, only train PLN
            # for name, param in self.model.named_parameters():
            #     if 'classifier' not in name:  # classifier layer
            #         param.learn = False
            #     else:
            #         param.learn = True

            print('----Task', idt[task_id], '----')
            for i in range(0, self.inner_update_step_eval):
                print('----Testing Inner Step ', i, '-----')
                all_loss = []
                for inner_step, batch in enumerate(support_dataloader):
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, attention_mask, segment_ids, label_id = batch
                    outputs = self.model(input_ids, attention_mask, segment_ids, labels=label_id)
                    inner_optimizer.zero_grad()
                    loss = outputs[0]
                    loss.backward()
                    inner_optimizer.step()
                    all_loss.append(loss.item())
                print("Inner Loss on support set: ", np.mean(all_loss))

            
            with torch.no_grad():
                print('----Testing Outer Step-----')
                self.model.eval()
                query_dataloader = DataLoader(query, sampler=None, batch_size=16)
                correct = 0
                total = 0
                for i, batch in enumerate(query_dataloader):
                    #query_batch = iter(query_dataloader).next()
                    query_batch = tuple(t.to(self.device) for t in batch)
                    q_input_ids, q_attention_mask, q_segment_ids, q_label_id = query_batch
                    q_outputs = self.model(q_input_ids, q_attention_mask, q_segment_ids, labels=q_label_id)
                    q_loss = q_outputs[0]
                    q_logits = F.softmax(q_outputs[1], dim=1)
                    pre_label_id = torch.argmax(q_logits, dim=1)
                    #pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
                    #q_label_id = q_label_id.detach().cpu().numpy().tolist()
                    
                    total += q_label_id.size(0)
                    correct += pre_label_id.eq(q_label_id.to(self.device).view_as(pre_label_id)).sum().item()
                acc = correct / total
                #acc = accuracy_score(pre_label_id, q_label_id)
                print('Outer Acc on query set: ', acc)
                del inner_optimizer
        
        # Test forgetting
        with torch.no_grad():
            print('----Testing Forgetting-----')
            for task_id, task in enumerate(batch_tasks):
                query = task[1]
                self.model.to(self.device)
                self.model.eval()
                correct = 0
                total = 0
                query_dataloader = DataLoader(query, sampler=None, batch_size=16)
                for i, batch in enumerate(query_dataloader):
                    query_batch = tuple(t.to(self.device) for t in batch)
                    q_input_ids, q_attention_mask, q_segment_ids, q_label_id = query_batch
                    q_outputs = self.model(q_input_ids, q_attention_mask, q_segment_ids, labels=q_label_id)
                    q_loss = q_outputs[0]
                    q_logits = F.softmax(q_outputs[1], dim=1)
                    pre_label_id = torch.argmax(q_logits, dim=1)
                    
                    total += q_label_id.size(0)
                    correct += pre_label_id.eq(q_label_id.to(self.device).view_as(pre_label_id)).sum().item()
                acc = correct / total
                task_accs.append(acc)
                print("accuracy on task " + str(task_id) + " after finalizing: " + str(acc))
                self.model.to(torch.device('cpu'))
            
            torch.cuda.empty_cache()
            gc.collect()

            return np.mean(task_accs)
