from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertModel, BertPreTrainedModel
from copy import deepcopy
import gc
import torch
from sklearn.metrics import accuracy_score
import numpy as np
# added 
from mutlitask_bert import BertForSpanClassification, BertForSequenceClassificationMultiTask
from transformers import glue_processors, superglue_processors
from scipy.stats import pearsonr



class Learner(nn.Module):
    """
    This is a modified version of `Meta Learner` from  `maml.py`. Instead of calling a `BertForSequenceClassification`,
    bert based uncased model is called and depends on the task outputs, a FC layer is added as top fine-tune layer to 
    generate results. Again as in Yogatama et al has pointed out, the fine-tune layer should be task specfic, rather than
    data (output class) specific. 
    """

    def __init__(self, args):
        """
        :param:
            
        """
        super(Learner, self).__init__()

        self.outer_batch_size = args.outer_batch_size
        self.inner_batch_size = args.inner_batch_size
        self.outer_update_lr = args.outer_update_lr
        self.inner_update_lr = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.inner_update_step_eval = args.inner_update_step_eval
        self.bert_model = args.bert_model
        self.meta_testing_size = args.meta_testing_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertModel.from_pretrained(self.bert_model)
        self.outer_optimizer = Adam(self.model.parameters(), lr=self.outer_update_lr)
        self.classifiers = []

    def get_ft_layer_loss(self, tasks_and_modes=tasks_and_modes, task_order=ids, current_id=task_id):
        '''
        helper function to return PLN and loss upon different tasks. 
        '''
        task, mode = tasks_and_modes[task_order[current_id]]
        if mode == 'classification':
            num_labels = 2
            loss_fn = CrossEntropyLoss()
        elif mode == 'regression':
            num_labels = 1
            loss_fn = MSELoss()
        else: 
            raise NotImplementedError('do not support current mode type', mode)
        # Random Initialize W_ for classification
        classifier = nn.Linear(768, num_labels).to(self.device)
        torch.nn.init.xavier_uniform_(classifier.weight.data)
        return classifier, loss_fn, mode
    
    def get_acc(self, probs, labels, mode=mode):
        if mode == 'classification':
            logits = torch.argmax(probs, dim=1)
            acc = accuracy_score(labels.cpu(), logits.cpu())
            return acc
        elif mode == 'regression':
            pears = pearsonr(logits.cpu(), labels.cpu())
            return pears

    def forward(self, ids, batch_tasks, tasks_and_modes):
        """
        meta-training mode, where only the BERT (RLN) is updated and saved
        fine-tune layer (PLN) is randomly initialized for each task and never saved.
        :param:
            ids: a list of task ids from arg.training_tasks
            tasks_and_modes: a list of task name and output modes
        :input:
            batch_tasks: numbers of (support, query) batches 

        batch_tasks = [(support TensorDataset, query TensorDataset),
                    (support TensorDataset, query TensorDataset),
                    (support TensorDataset, query TensorDataset),
                    (support TensorDataset, query TensorDataset)]

        # support = query = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids) for glue, others please check `task_all.py`
        """
        task_accs = []
        num_task = len(batch_tasks)
        print(ids)
        for task_id, task in enumerate(batch_tasks):
            support, query = task

            classifier, loss_fn, mode = get_ft_layer_loss(tasks_and_modes=tasks_and_modes, task_order=ids, current_id=task_id)

            inner_optimizer = Adam(classifier.parameters(), lr=self.inner_update_lr)

            support_dataloader = DataLoader(support, sampler=RandomSampler(support),
                                            batch_size=self.inner_batch_size)
            
            # inner step of meta-learning: train on classifier (PLN) only
            self.model.to(self.device)
            self.model.eval()
            classifier.requires_grad(True)
            print('----Task', ids[task_id], '----')
            for i in range(0, self.inner_update_step):
                print('----Training Inner Step ', i, '-----')
                all_loss = []
                for inner_step, batch in enumerate(support_dataloader):

                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, attention_mask, segment_ids, label_id = batch
                    outputs_hidden = self.model(input_ids, attention_mask, segment_ids)
                    output_digits = classifier(outputs_hidden)
                    loss = loss_fn(output_digits.view(-1, num_labels), label_id.view(-1)) 
                    # backward
                    inner_optimizer.zero_grad()
                    loss.backward()
                    inner_optimizer.step()
                    # log
                    all_loss.append(loss.item())

                # if i % 4 == 0:
                print("Loss in support set: ", np.mean(all_loss))

            print('----Training Outer Step-----')
            # outer step of meta-learning: train on bert only
            self.model.train()
            self.classifiers.append(classifier)
            classifier.eval()

            query_dataloader = DataLoader(query, sampler=None, batch_size=len(query))
            query_batch = iter(query_dataloader).next()
            query_batch = tuple(t.to(self.device) for t in query_batch)
            q_input_ids, q_attention_mask, q_segment_ids, q_label_id = query_batch
            q_outputs_hidden = self.model(q_input_ids, q_attention_mask, q_segment_ids)
            q_output_logits = classifier(q_outputs_hidden)
            # backward step
            self.outer_optimizer.zero_grad()
            q_loss = loss_fn(q_output_logits.view(-1, num_labels), q_label_id.view(-1)) 
            q_loss.backward()
            self.outer_optimizer.step()
            acc = get_acc(probs=q_output_logits, labels=q_label_id, mode=mode)
            task_accs.append(acc)
            print('Acc/Pearson in query set: ', acc)

            del inner_optimizer
            torch.cuda.empty_cache()
            gc.collect()
            self.model.to(torch.device('cpu'))

        return np.mean(task_accs)

    def finetune(self, idt, batch_tasks, tasks_and_modes):
        '''
        during meta-testing phase, inner loop update on both BERT (RLN) and classifier (PLN)
        while outer loop do not update on anything, rather checking on how well our model has learnt 
        '''
        task_accs = []
        num_task = len(batch_tasks)
        for task_id, task in enumerate(batch_tasks):
            support, query = task
            classifier, loss_fn, mode = get_ft_layer_loss(tasks_and_modes=tasks_and_modes, task_order=ids, current_id=task_id)    

            inner_optimizer = Adam(classifier.parameters(), lr=self.inner_update_lr)
            
            support_dataloader = DataLoader(support, sampler=RandomSampler(support, replacement=True, num_samples = self.meta_testing_size),
                                            batch_size=self.inner_batch_size)
\
            self.model.to(self.device)
            self.model.train()
            classifier.train()
            print('----Task', idt[task_id], '----')
            for i in range(0, self.inner_update_step_eval):
                print('----Testing Inner Step ', i, '-----')
                all_loss = []
                for inner_step, batch in enumerate(support_dataloader):
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, attention_mask, segment_ids, label_id = batch
                    outputs_hidden = self.model(input_ids, attention_mask, segment_ids)
                    output_logits = classifier(outputs_hidden)
                    loss = loss_fn(output_logits.view(-1, num_labels), label_id)
                    # pre_label_id = torch.argmax(output_logits, dim=1)
                    self.inner_optimizer.zero_grad()
                    self.outer_optimizer.zero_grad()
                    loss.backward()
                    inner_optimizer.step()
                    outer_optimizer.step()
                    all_loss.append(loss.item())
                print("Inner Loss on support set: ", np.mean(all_loss))

            
            with torch.no_grad():
                print('----Testing Outer Step-----')
                self.model.eval()
                classifier.requires_grad(False)
                query_dataloader = DataLoader(query, sampler=None, batch_size=16)
                correct = 0
                total = 0
                num_batch = 0
                total_acc = 0
                for i, batch in enumerate(query_dataloader):
                    #query_batch = iter(query_dataloader).next()
                    query_batch = tuple(t.to(self.device) for t in batch)
                    q_input_ids, q_attention_mask, q_segment_ids, q_label_id = query_batch
                    q_outputs_hidden = self.model(q_input_ids, q_attention_mask, q_segment_ids)
                    q_output_logits = classifier(q_outputs_hidden)
                    q_loss = loss_fn(q_output_logits.view(-1, num_labels), q_label_id)

                    # org
                    pre_label_id = torch.argmax(q_output_logits, dim=1)
                    total += q_label_id.size(0)
                    correct += pre_label_id.eq(q_label_id.to(self.device).view_as(pre_label_id)).sum().item()
                    # mine
                    acc_f = get_acc(probs=q_output_logits, labels=q_label_id, mode=mode)
                    total_acc += acc_f
                    num_batch += 1
                acc = correct / total) # org
                print('Outer Acc on query set: ', acc)
                print('Mine Outer Acc on query set:' total_acc / num_batch)
                del inner_optimizer
        
        # Test forgetting, none of the bert or classifier should be updated
        # TODO: need to append all the classifier together and look at the performance of RLN layer
        with torch.no_grad():
            print('----Testing Forgetting-----')
            for task_id, task in enumerate(batch_tasks):
                query = task[1]
                self.model.to(self.device)
                self.model.eval()
                classifier.requires_grad(False)
                correct = 0
                total = 0
                query_dataloader = DataLoader(query, sampler=None, batch_size=16)
                for i, batch in enumerate(query_dataloader):
                    query_batch = tuple(t.to(self.device) for t in batch)
                    q_input_ids, q_attention_mask, q_segment_ids, q_label_id = query_batch

                    q_outputs_hidden = self.model(q_input_ids, q_attention_mask, q_segment_ids)
                    q_output_logits = classifier(q_outputs_hidden)
                    q_loss = loss_fn(q_output_logits.view(-1, num_labels), q_label_id)
                    pre_label_id = torch.argmax(q_output_logits, dim=1)
                    
                    total += q_label_id.size(0)
                    correct += pre_label_id.eq(q_label_id.to(self.device).view_as(pre_label_id)).sum().item()
                acc = correct / total
                task_accs.append(acc)
                print("accuracy on task " + str(task_id) + " after finalizing: " + str(acc))
                self.model.to(torch.device('cpu'))
            
            torch.cuda.empty_cache()
            gc.collect()

            return np.mean(task_accs)
