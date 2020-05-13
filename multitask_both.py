from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertModel, BertPreTrainedModel, glue_tasks_num_labels, superglue_tasks_num_labels
from copy import deepcopy
import gc
import torch
from sklearn.metrics import accuracy_score
import numpy as np

# change
from transformers import glue_processors, superglue_processors
from scipy.stats import pearsonr
from tqdm import tqdm

class bilinear_classifier(nn.Module):

    def __init__(self, num_labels = 2, input_size = 768):
        super(bilinear_classifier, self).__init__()
        self.bilinear_layer = nn.Bilinear(input_size, input_size, num_labels)
        #self.dropout = nn.Dropout(0.1, inplace = False)
        self.output = nn.Sigmoid()

    def forward(self, span_emb_1, span_emb_2, label=None):
        x = self.bilinear_layer(span_emb_1, span_emb_2)
        x = self.output(x)
        return x



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
        self.inner_batch_size = args.inner_batch_size if not args.oml else 1
        self.outer_update_lr = args.outer_update_lr
        self.inner_update_lr = args.inner_update_lr
        self.inner_update_step = args.inner_update_step if not args.oml else 1
        self.inner_update_step_eval = args.inner_update_step_eval
        self.bert_model = args.bert_model
        self.meta_testing_size = args.meta_testing_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertModel.from_pretrained(self.bert_model)
        self.outer_optimizer = Adam(self.model.parameters(), lr=self.outer_update_lr)
        self.classifiers = {}
        self.modes = {**glue_tasks_num_labels, **superglue_tasks_num_labels}

    def init_weights(self, m):
        if type(m) in [nn.Linear, nn.Bilinear]:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def get_ft_layer_loss(self, task_order, current_id):
        '''
        helper function to return PLN and loss upon different tasks. 
        '''
        task = task_order[current_id]

        try: 
            num_labels = self.modes[task]
        except:
            raise NotImplementedError ('task not supported, supported tasks are', self.modes)

        if num_labels == 1:
            loss_fn = MSELoss()
        else: 
            loss_fn = CrossEntropyLoss()

        # Random Initialize W_ for classification
        # nn.Sequential does not support multiple inputs yet 
        if task in ['copa','wic', 'wsc']: 
            ft_layer = bilinear_classifier(num_labels = num_labels, input_size = 768)
        else:
            ft_layer = nn.Sequential(nn.Dropout(p=0.1, inplace = False),
                    nn.Linear(768, num_labels)).to(self.device)
        ft_layer.apply(self.init_weights)
        return ft_layer, loss_fn, num_labels
    
    def get_acc(self, probs, labels, num_labels, normalize=True):
        '''
            given prediction & labels, return corresponding loss base on number of labels
        '''
        if num_labels == 1:
            pears = pearsonr(probs.view(-1).cpu(), labels.view(-1).cpu())[0]
            return pears
        else:
            logits = torch.argmax(probs, dim=1)
            acc = accuracy_score(labels.view(-1).cpu(), logits.view(-1).cpu(), normalize)
            return acc

    def get_loss(self, batch, base_model, classifier, current_task, loss_fn, verbose = False, return_acc = False):
        '''
            given RLN and PLN, generate logits base on current task.
        '''
        if current_task == 'copa':
            input_ids_1, attention_mask_1, segment_ids_1, label_id,\
            input_ids_2, attention_mask_2, segment_ids_2, _ = batch
            if verbose: print('input shapes', input_ids_1.shape, attention_mask_1.shape, segment_ids_1.shape)
            outputs_hidden_1 = base_model(input_ids_1, attention_mask_1, segment_ids_1)[1]
            outputs_hidden_2 = base_model(input_ids_2, attention_mask_2, segment_ids_2)[1]
            if verbose: print('outputs hidden shapes',outputs_hidden_1.shape, outputs_hidden_2.shape)
            output_logits = classifier(outputs_hidden_1, outputs_hidden_2)
        elif current_task in ['wic', 'wsc']:
            all_input_ids, all_attention_mask, all_token_type_ids, label_id = batch[:4]
            all_span_1_mask, all_span_1_text, all_span_2_mask, all_span_2_text = batch[4:]
            outputs_hidden = base_model(all_input_ids, all_attention_mask, all_token_type_ids)[0]
            prod_1 = (outputs_hidden * all_span_1_mask.unsqueeze(-1).float()).sum(dim=-2)
            prod_2 = (outputs_hidden * all_span_2_mask.unsqueeze(-1).float()).sum(dim=-2)
            output_logits = classifier(prod_1, prod_2)
        else:
            input_ids, attention_mask, segment_ids, label_id = batch
            outputs_hidden = base_model(input_ids, attention_mask, segment_ids)[1]
            output_logits = classifier(outputs_hidden)
        if current_task == 'sts-b': output_logits = output_logits.view(-1)
        loss = loss_fn(output_logits, label_id.view(-1)) 
        if return_acc:
            return loss, output_logits, label_id
        else: 
            return loss

    def forward(self, ids, batch_tasks):
        """
        meta-training mode, where only the BERT (RLN) is updated and saved
        fine-tune layer (PLN) is randomly initialized for each task and never saved.
        :param:
            ids: a list of task NAMEs from arg.training_tasks
        :input:
            batch_tasks: numbers of (support, query) batches 

        batch_tasks = [(support TensorDataset, query TensorDataset),
                    (support TensorDataset, query TensorDataset),
                    (support TensorDataset, query TensorDataset),
                    (support TensorDataset, query TensorDataset)]

        # support = query = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids) for glue tasks, others please check speficially in `task_all.py`
        """
        task_accs = []
        num_task = len(batch_tasks)
        print(ids)
        for task_id, task in enumerate(batch_tasks):
            support, query = task

            classifier, loss_fn, num_labels = self.get_ft_layer_loss(task_order=ids, current_id=task_id)

            inner_optimizer = Adam(classifier.parameters(), lr=self.inner_update_lr)

            support_dataloader = DataLoader(support, sampler=RandomSampler(support),
                                            batch_size=self.inner_batch_size)
            
            # inner step of meta-learning: train on classifier (PLN) only
            self.model.to(self.device)
            self.model.eval()
            classifier.requires_grad_(True)
            current_task = ids[task_id]
            print('----Task', current_task, '----')
            for i in range(0, self.inner_update_step):
                print('----Training Inner Step ', i, '-----')
                all_loss = []
                for inner_step, batch in enumerate(support_dataloader):

                    batch = tuple(t.to(self.device) for t in batch)
                    loss = self.get_loss(batch=batch, base_model=self.model, classifier=classifier, current_task=current_task, loss_fn = loss_fn)
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
            classifier.eval()

            query_dataloader = DataLoader(query, sampler=None, batch_size=len(query))
            query_batch = iter(query_dataloader).next()
            query_batch = tuple(t.to(self.device) for t in query_batch)
            q_loss, q_output_logits, q_label_id = self.get_loss(batch=query_batch, base_model=self.model, 
                                    classifier=classifier, current_task=current_task, loss_fn = loss_fn, return_acc = True)
            # backward step
            self.outer_optimizer.zero_grad()
            q_loss.backward()
            self.outer_optimizer.step()
            acc = self.get_acc(probs=q_output_logits, labels=q_label_id, num_labels=num_labels)
            task_accs.append(acc)
            print('Acc in query set: ', acc)

            del inner_optimizer
            torch.cuda.empty_cache()
            gc.collect()
            self.model.to(torch.device('cpu'))

        return np.mean(task_accs)

    def finetune(self, idt, batch_tasks):
        '''
        during meta-testing phase, inner loop update on both BERT (RLN) and classifier (PLN)
        while outer loop do not update on anything, rather checking on how well our model has learnt 
        :param:
            idt: list of input tasks NAMES for metat-testing
        :input:
            batch_tasks: batch of tasks
        '''
        task_accs = []
        num_task = len(batch_tasks)
        for task_id, task in enumerate(batch_tasks):
            support, query = task
            current_task = idt[task_id]
            classifier, loss_fn, num_labels = self.get_ft_layer_loss(task_order=idt, current_id=task_id)    
            inner_optimizer = Adam(classifier.parameters(), lr=self.inner_update_lr)
            
            support_dataloader = DataLoader(support, sampler=RandomSampler(support, replacement=True, num_samples = self.meta_testing_size), batch_size=self.inner_batch_size)

            self.model.to(self.device)
            self.model.train()
            classifier.train()
            print('----Task', idt[task_id], '----')
            for i in range(0, self.inner_update_step_eval):
                print('----Testing Inner Step ', i, '-----')
                all_loss = []
                for inner_step, batch in enumerate(support_dataloader):
                    batch = tuple(t.to(self.device) for t in batch)
                    # forward
                    loss = self.get_loss(batch=batch, base_model=self.model, 
                        classifier=classifier, current_task=current_task, loss_fn = loss_fn)
                    # backward
                    inner_optimizer.zero_grad()
                    self.outer_optimizer.zero_grad()
                    loss.backward()
                    inner_optimizer.step()
                    self.outer_optimizer.step()
                    # log
                    all_loss.append(loss.item())
                print("Inner Loss on support set: ", np.mean(all_loss))

            
            with torch.no_grad():
                print('----Testing Outer Step-----')
                self.model.eval()
                classifier.requires_grad_(False)
                query_dataloader = DataLoader(query, sampler=None, batch_size=16)
    
                total = 0
                total_acc = 0
                for i, batch in enumerate(query_dataloader):
                    #query_batch = iter(query_dataloader).next()
                    query_batch = tuple(t.to(self.device) for t in batch)
                    q_loss, q_output_logits, q_label_id = self.get_loss(batch=query_batch, base_model=self.model, 
                        classifier=classifier, current_task=current_task, loss_fn = loss_fn, return_acc = True)
                    acc = self.get_acc(probs=q_output_logits, labels=q_label_id, num_labels = num_labels, normalize=False)
                    total_acc += acc
                    total += q_label_id.size(0)
                print('Outer Acc on query set:', total_acc / total)
                del inner_optimizer
            self.classifiers[current_task] = classifier
        
        # Test forgetting, none of the bert or classifier should be updated
        with torch.no_grad():
            print('----Testing Forgetting-----')
            for task_id, task in enumerate(batch_tasks):
                current_task = idt[task_id]
                nums_labels = self.modes[current_task]
                loss_fn = MSELoss() if nums_labels == 1 else CrossEntropyLoss()
                query = task[1]

                self.model.to(self.device)
                self.model.eval()
                classifier = self.classifiers[current_task] # recall the best PLN trainied on meta-testing inner loop phase
                classifier.eval()

                total = 0
                total_acc = 0
                query_dataloader = DataLoader(query, sampler=None, batch_size=16)
                for i, batch in enumerate(query_dataloader):
                    query_batch = tuple(t.to(self.device) for t in batch)
                    q_loss, q_output_logits, q_label_id = self.get_loss(batch=query_batch, base_model=self.model, 
                                        classifier=classifier, current_task=current_task, loss_fn = loss_fn, return_acc = True)
                    acc = self.get_acc(probs=q_output_logits, labels=q_label_id, num_labels = nums_labels, normalize=False)
                    total_acc += acc
                    total += q_label_id.size(0)
                    acc = total_acc / tota
                task_accs.append(acc)
                print("accuracy on task " + current_task + " after finalizing: " + str(acc))
                self.model.to(torch.device('cpu'))
            
            torch.cuda.empty_cache()
            gc.collect()

            return np.mean(task_accs)
