# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import torch
from torch.utils.data import Dataset
import numpy as np
import collections
import random
import json, pickle
from torch.utils.data import TensorDataset, RandomSampler
from transformers import glue_processors as processors
from transformers import glue_output_modes as output_modes
from transformers import glue_convert_examples_to_features as convert_examples_to_features
import logging

# NOTE: Before running this script, please makes sure all 8 GLUE datasets are downloaded 
# in local by running python3 ../../utils/download_glue_data.py under transformers directory


## TODO: 
## 1. in arguments add 'data_dir, model_name_or_path (removed), max_seq_length, local_rank' done
## 2. add SQuAD and SUPERGlue
## 3. Optimize run time? current function seperate datasets and select k samples randomly. done
##    Run time is improved by without preprocessing examples to features and recall features everytime,
##    instead this script calls k examples randomly and transform features in place each time. 

# USAGE: 
# INPUT:
# ORG:  test = MetaTask(test_examples, num_task = args.num_task_test, k_support=args.k_spt, 
#                    k_query=args.k_qry, tokenizer = tokenizer)

# NOW:  test = MetaTask(num_task = args.num_task_test, k_support=args.k_spt, 
#                    k_query=args.k_qry, tokenizer = tokenizer, True)


# OUTPUT: ex batch = 4
    # batch = [(support TensorDataset, query TensorDataset),
    #          (support TensorDataset, query TensorDataset),
    #          (support TensorDataset, query TensorDataset),
    #          (support TensorDataset, query TensorDataset)]
        
    # support = query =  TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids)
logger = logging.getLogger(__name__)
class MetaTask(Dataset):
    ''' 
    Modified MetaTask takes all GLUE tasks, namely cola, mnli, mnli-mm, mrpc, sst-2, sts-b, 
    qqp, qnli, rte and wnli and convert them from raw test into features. 
    '''
    
    def __init__(self, args, num_task, k_support, k_query, tokenizer, max_seq_length, evaluate=False):
        """
        :param num_task: number of training tasks.
        :param k_support: number of support sample per task
        :param k_query: number of query sample per task
        :param tokenizer: tokenizer uses to tokenzie from word to sequence
        :param max_seq_length: length of the tokenzier vector
        :param evaluate: indicate whether the dataset is from training/ evaluate sets
        """

        self.num_task        = num_task
        self.k_support       = k_support
        self.k_query         = k_query
        self.tokenizer       = tokenizer
        self.max_seq_length  = max_seq_length
        self.evaluate        = evaluate
        self.local_rank      = args.local_rank
        self.data_dir        = args.data_dir
        self.bert_model      = args.bert_model
        self.overwrite_cache = args.overwrite_cache

        self.create_batch(self.num_task)

    def create_batch(self, num_task):
        '''
        Randomly select number of examples from each task into supports (meta training dataset) and queries (meta evaluating dataset)
        '''
        self.supports = []  # support set
        self.queries = []  # query set
        # 1. randomly select num_task GLUE tasks 
        if len(processors.keys) < num_task:
            logger.info('Num of tasks exceed avaliable tasks, drawing tasks with replacement')
            tasks = random.choices(list(processors.keys()), k = num_task)
        else:
            tasks = random.sample(list(processors.keys()), num_task) # select k unique tasks
 
        for b in range(num_task):  ## for each task
            task = tasks[b]
            #print(task)
            # 2.select k_support + k_query examples from task randomly
            dataset = self.load_and_cache_examples(task, self.tokenizer, self.evaluate) # map style dataset 

            exam_train = dataset[:self.k_support]
            exam_test  = dataset[self.k_support:]

            # 3. put into support and queries 
            self.supports.append(exam_train)
            self.queries.append(exam_test)

    def load_and_cache_examples(self, task, tokenizer, evaluate=False):
        '''
        Heavily insipired from official loading and cache scripts from Huggingface Transformer func load_and_cache_examples
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
        print(cached_downloaded_file)

        logger.info(f"Creating {self.k_support+self.k_query} features from dataset file at {cached_downloaded_file}")
        label_list = processor.get_labels()

        examples = (
                processor.get_dev_examples(cached_downloaded_file) if evaluate else processor.get_train_examples(cached_downloaded_file)
            )

        if len(selected_examples) < self.k_query + self.k_support:
            selected_examples = random.choices(examples, k = self.k_support + self.k_query)
        else:
            selected_examples = random.sample(examples, self.k_support + self.k_query)

        features = convert_examples_to_features(
            selected_examples, tokenizer, max_length=self.max_seq_length, label_list=label_list, output_mode=output_mode,
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
        support_set = self.supports[index]
        query_set   = self.queries[index]
        return support_set, query_set

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.num_task
