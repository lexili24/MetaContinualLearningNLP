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
        self.inner_batch_size = args.inner_batch_size
        self.outer_update_lr = args.outer_update_lr
        self.inner_update_lr = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.inner_update_step_eval = args.inner_update_step_eval
        self.bert_model = args.bert_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = BertForSequenceClassification.from_pretrained(self.bert_model, num_labels=self.num_labels)
        self.outer_optimizer = Adam(self.model.parameters(), lr=self.outer_update_lr)
        self.model.train()

    def forward(self, ids, batch_tasks):
        """
        batch = [(support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset)]

        # support = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids)
        """
        task_accs = []
        num_task = len(batch_tasks)
        print(ids)
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
            # Freeze the representation layers of the model to train the inner loop
            # for param in fast_model.bert.embeddings.parameters():
            #     param.requires_grad = False
            # Another method to change the freezing layer

            for name, param in self.model.named_parameters():
                if 'classifier' not in name:  # classifier layer
                    param.learn = False
                else:
                    param.learn = True

            print('----Task', ids[task_id], '----')
            for i in range(0, self.inner_update_step):
                if i % 1 == 0:
                    print('----Training Inner Step ', i, '-----')
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

                    # if i % 4 == 0:
                    print("Inner Loss: ", np.mean(all_loss))

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

            self.outer_optimizer.zero_grad()
            q_loss = q_outputs[0]
            q_loss.backward()
            self.outer_optimizer.step()

            q_logits = F.softmax(q_outputs[1], dim=1)
            pre_label_id = torch.argmax(q_logits, dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = q_label_id.detach().cpu().numpy().tolist()

            acc = accuracy_score(pre_label_id, q_label_id)
            task_accs.append(acc)
            print('Outer Acc: ', acc)

            del inner_optimizer
            torch.cuda.empty_cache()

            gc.collect()

        return np.mean(task_accs)

    def finetune(self, idt, batch_tasks):
        """
        batch = [(support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset)]

        # support = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids)
        """
        task_accs = []
        num_task = len(batch_tasks)
        for task_id, task in enumerate(batch_tasks):
            support = task[0]
            query = task[1]

            self.model.to(self.device)
            self.model.train()
            support_dataloader = DataLoader(support, sampler=RandomSampler(support),
                                            batch_size=self.inner_batch_size)

            inner_optimizer = Adam(self.model.parameters(), lr=self.inner_update_lr)

            for name, param in self.model.named_parameters():
                if 'classifier' not in name:  # classifier layer
                    param.learn = False
                else:
                    param.learn = True

            print('----Task', idt[task_id], '----')
            for i in range(0, self.inner_update_step_eval):
                if i % 1 == 0:
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

                    # if i % 4 == 0:
                    print("Inner Loss: ", np.mean(all_loss))

            print('----Testing Outer Step-----')
            self.model.eval()
            query_dataloader = DataLoader(query, sampler=None, batch_size=len(query))
            query_batch = iter(query_dataloader).next()
            query_batch = tuple(t.to(self.device) for t in query_batch)
            q_input_ids, q_attention_mask, q_segment_ids, q_label_id = query_batch

            q_outputs = self.model(q_input_ids, q_attention_mask, q_segment_ids, labels=q_label_id)

            q_loss = q_outputs[0]
            q_logits = F.softmax(q_outputs[1], dim=1)
            pre_label_id = torch.argmax(q_logits, dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = q_label_id.detach().cpu().numpy().tolist()

            acc = accuracy_score(pre_label_id, q_label_id)
            task_accs.append(acc)
            print('Outer Acc: ', acc)

            del inner_optimizer
            torch.cuda.empty_cache()

            gc.collect()

        # Test forgetting
        print('----Testing Forgetting-----')
        for task_id, task in enumerate(batch_tasks):
            query = task[1]
            self.model.to(self.device)
            query_dataloader = DataLoader(query, sampler=None, batch_size=len(query))
            query_batch = iter(query_dataloader).next()
            query_batch = tuple(t.to(self.device) for t in query_batch)
            q_input_ids, q_attention_mask, q_segment_ids, q_label_id = query_batch

            q_outputs = self.model(q_input_ids, q_attention_mask, q_segment_ids, labels=q_label_id)

            q_loss = q_outputs[0]
            q_logits = F.softmax(q_outputs[1], dim=1)
            pre_label_id = torch.argmax(q_logits, dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = q_label_id.detach().cpu().numpy().tolist()

            acc = accuracy_score(pre_label_id, q_label_id)
            print("accuracy on task " + str(task_id) + " after finalizing: " + str(acc))

        return np.mean(task_accs)
