import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from transformers import BertModel, BertTokenizer
from torch.utils.data import TensorDataset
from transformers import glue_processors as processors
from transformers import glue_output_modes as output_modes
from transformers import glue_convert_examples_to_features as convert_examples_to_features
import logging
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from transformers import BertForSequenceClassification
from copy import deepcopy
import gc
from sklearn.metrics import accuracy_score
import torch
import argparse

class BertTask_Baseline(Dataset):
    ''' 
    Before running this script, please makes sure all 8 GLUE datasets are downloaded in local by running python3 ../../utils/download_glue_data.py
    Modified MetaTask takes all 10 GLUE tasks, namely cola, mnli, mnli-mm, mrpc, sst-2, sts-b, 
    qqp, qnli, rte and wnli and convert them from raw test into features. 
    '''
    
    def __init__(self, args, tokenizer, max_seq_length, task, evaluate=False,sample=False):
        """
        :param num_task: number of training tasks.
        :param tokenizer: tokenizer uses to tokenzie from word to sequence
        :param max_seq_length: length of the tokenzier vector
        :param evaluate: indicate whether the dataset is from training/ evaluate sets
        """

        self.tokenizer       = tokenizer
        self.max_seq_length  = max_seq_length
        self.evaluate        = evaluate
        self.local_rank      = args.local_rank
        self.data_dir        = args.data_dir
        self.bert_model      = args.bert_model
        self.overwrite_cache = args.overwrite_cache
        self.sample = sample
        self.task = task
        self.create_batch()
        
        
    def create_batch(self):
        '''
        Randomly select number of examples from each task into supports (meta training dataset) and queries (meta evaluating dataset)
        '''
        
        # 1. randomly select num_task GLUE tasks 
        task = self.task 

        self.dataset = self.load_and_cache_examples(task, self.tokenizer, self.evaluate, self.sample) # map style dataset 


    def load_and_cache_examples(self, task, tokenizer, evaluate=False, sample=False):
        '''
        Copied from official loading and cache scripts from Huggingface Transformer load_and_cache_examples
        https://github.com/huggingface/transformers/blob/master/examples/run_glue.py#L334
        '''
        folder_name = {'cola': 'CoLA', 'mnli-mm':'MNLI'}
        if task in folder_name:
            task_data_path = folder_name[task]
        else:
            task_data_path = task.upper()


        if self.local_rank not in [-1, 0] and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        processor = processors[task]()
        output_mode = output_modes[task]
        cached_downloaded_file = os.path.join(self.data_dir, task_data_path)
        # print(cached_downloaded_file)

        logger.info(f"Creating features from dataset file at {cached_downloaded_file}")
        label_list = processor.get_labels()

        examples = (
                processor.get_dev_examples(cached_downloaded_file) if evaluate else processor.get_train_examples(cached_downloaded_file)
            )
        if sample:
            examples = random.sample(examples, sample)

        features = convert_examples_to_features(
            examples, tokenizer, max_length=self.max_seq_length, label_list=label_list, output_mode=output_mode,
        )

        if self.local_rank == 0 and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        return dataset

    def __getitem__(self, index):
        dataset_set = self.dataset[index]
        return dataset_set

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return len(self.dataset)


class Bert_trainer(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        """
        :param args:
        """
        super(Bert_trainer, self).__init__()

        self.num_labels = args.num_labels
        self.batch_size = args.batch_size
        self.update_lr  = args.update_lr
    

        self.bert_model = args.bert_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = BertForSequenceClassification.from_pretrained(self.bert_model, num_labels = self.num_labels)
        self.outer_optimizer = Adam(self.model.parameters(), lr=self.update_lr)
        self.model.train()

    def forward(self, datasets,training=True):
        """
        batch_tasks = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids)
        """

        num_task = len(datasets)
        self.model.to(self.device)
        acc = None

        
        if training:
            dataloader = DataLoader(datasets,batch_size=self.batch_size)
            for data in dataloader:
                all_loss = []
                batch = tuple(t.to(self.device) for t in data)
                input_ids, attention_mask, segment_ids, label_id = batch
                outputs = self.model(input_ids, attention_mask, segment_ids, labels = label_id)
                loss = outputs[0]              
                loss.backward()
                self.outer_optimizer.step()
                self.outer_optimizer.zero_grad()
                all_loss.append(loss.item())
            self.model.to(torch.device('cpu'))
            return outputs
        else:
            
            with torch.no_grad():
                correct = 0
                total = 0
                self.model.to(torch.device(self.device))
                dataloader = DataLoader(datasets,batch_size=self.batch_size)
                for data in dataloader:
                    
                    query_batch = iter(dataloader).next()
                    query_batch = tuple(t.to(self.device) for t in query_batch)
                    q_input_ids, q_attention_mask, q_segment_ids, q_label_id = query_batch
                    q_outputs = self.model(q_input_ids, q_attention_mask, q_segment_ids, labels = q_label_id)
                
                    q_logits = F.softmax(q_outputs[1],dim=1)
                    pre_label_id = torch.argmax(q_logits,dim=1)
                    # pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
                    # label_id = q_label_id.detach().cpu().numpy().tolist()
                    total += q_label_id.size(0)
                    correct += pre_label_id.eq(q_label_id.to(self.device).view_as(pre_label_id)).sum().item()
                acc = correct/total
                self.model.to(torch.device('cpu'))
        return acc

def main():
    logger=logging.getLogger()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task."
    )
    parser.add_argument(
        "--bert_model",
        default='bert-base-uncased',
        type=str,
        required=True,
        help="The type of bert model"
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--num_labels",default=2,type=int,help="Number of classes in classifications",)
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size training.")
    parser.add_argument("--train_sample_per_task", 
                        default=None, type=int, 
                        help="Number of Samples for each task. None for the whole data set")
    parser.add_argument("--eval_sample_per_task", 
                        default=None, type=int, 
                        help="Number of Samples for each evaling task. None for the whole data set")
    parser.add_argument("--update_lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--epochs", default=3, type=int, help="The epochs trained for each task.")
    parser.add_argument("--output_dir",default="bert_models&results",type=str,help="The output folder.")
    
    args = parser.parse_args()
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
    task_lists = ["cola", "sst-2", "mrpc","qqp","qnli","rte"]

    my_Bert = Bert_trainer(args)
    acc_results = []
    saving_path = os.path.join(args.output_dir,"model")
    if  not os.path.exists(saving_path):
        os.makedirs(saving_path)

    for i,task in enumerate(task_lists):
        train_data = BertTask_Baseline(args, tokenizer,128,task,sample=args.train_sample_per_task)
        print("Training on the {} task".format(task))
        for epoch in range(args.epochs):
            outputs = my_Bert(train_data)
        print("_____***Saving Model***___{}".format(task))
        torch.save(my_Bert.state_dict(), os.path.join(saving_path,"{}_params.pkl".format(task)))
        
        ### Evaluating
        accs = []
        for j in range(i+1):
            eval_task = task_lists[j]
            print("_____Evalating on the {}".format(eval_task))
            eval_data = BertTask_Baseline(args, tokenizer,128,eval_task,evaluate=True,sample=args.eval_sample_per_task)
            acc = my_Bert(eval_data,training=False)
            accs.append(acc)
            del eval_data
            _ = gc.collect()
            print("_____Finishing evalating on the {}".format(eval_task))
        acc_results.append(accs)
        print("Finishing training on the {} task".format(task))
        del train_data
        _ = gc.collect()

    acc_results_pad = [line+[""]*(6-len(line)) for line in acc_results]
    final_acc = "\n".join([",".join(list(map(str,line))) for line in acc_results_pad])
    
    with open(os.path.join(args.output_dir,"results.txt"),"w") as f:
        f.write(",".join(task_lists))
        f.write("\n")
        f.write(final_acc)


        
        
if __name__ == "__main__":
    main()

