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


# https://github.com/nyu-mll/jiant/blob/11e3b696c088260d138d9c75a851d2234a3cdb2f/src/models.py#L523 
# multi task model for reference 

#import pdb
import os
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import collections
import random
import json, pickle
from torch.utils.data import TensorDataset, RandomSampler, random_split
from transformers import glue_processors, superglue_processors
from transformers import glue_output_modes, superglue_output_modes
from transformers import squad_convert_examples_to_features
from transformers import glue_convert_examples_to_features, superglue_convert_examples_to_features
import logging
from transformers.data.processors.squad import SquadV2Processor

# NOTE: Before running this script, please makes sure all 8 GLUE datasets are downloaded 
# in local by running python3 ../../utils/download_glue_data.py under transformers directory
# download SQuAD at https://rajpurkar.github.io/SQuAD-explorer/
# download superglue using `download_superglue.py` and save it to same directory as GLUE data, in a sub-directory called `superglue`

# USAGE: 
# INPUT:
# ORG:  test = MetaTask(test_examples, num_task = args.num_task_test, k_support=args.k_spt, 
#                    k_query=args.k_qry, tokenizer = tokenizer)

# NOW:  test = MetaTask(num_task = args.num_task_test, k_support=args.k_spt, 
#                    k_query=args.k_qry, tokenizer = tokenizer, evaluate = True)


# OUTPUT: ex batch = 4
    # batch = [(support TensorDataset, query TensorDataset),
    #          (support TensorDataset, query TensorDataset),
    #          (support TensorDataset, query TensorDataset),
    #          (support TensorDataset, query TensorDataset)]
    
    # - Glue and Super-glue BoolQ, RTE2 and CB
    # train/test support = train/test query =  TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids)
    # - SQuaD:
    # train/test support = train/test query = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_start_positions,
    #                                                       all_end_positions, all_cls_index, all_p_mask, all_is_impossible, )
    # - Super-gLue COPA: 
    # train/test support = train/test query =  (TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids), 
    #                                           TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids) )
    # - Super-glue WiC and WSC: 
    # train/test support = train/test query = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids, 
    #                                                       all_span_1_mask, all_span_1_text, all_span_2_mask, all_span_2_text)
    
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

        self.num_task_train   = num_task
        self.k_support        = k_support
        self.k_query          = k_query
        self.tokenizer        = tokenizer
        self.max_seq_length   = max_seq_length
        self.evaluate         = evaluate
        self.local_rank       = args.local_rank
        self.data_dir         = args.data_dir
        self.bert_model       = args.bert_model

        self.doc_stride       = args.doc_stride
        self.max_query_length = args.max_query_length
        self.training_tasks   = args.training_tasks
        self.testing_tasks    = args.testing_tasks
        self.evaluate_whole   = args.evaluate_whole_set
        self.meta_testing_size= args.meta_testing_size
        self.tasks            = []
        self.labels           = []
        self.create_batch(self.num_task_train)

    def create_batch(self, train_num_task):
        '''
        Randomly select number of examples from each task into supports (meta training dataset) and queries (meta evaluating dataset)
        '''
        self.supports = []  # support set
        self.queries = []  # query set
        # 1. randomly select num_task GLUE + SuperGlue tasks

        if not self.evaluate: # training
            if len(self.training_tasks) < train_num_task:
                logger.info('Num of tasks exceed avaliable tasks, drawing tasks with replacement')
                self.tasks = random.choices(self.training_tasks, k = train_num_task)
            else:
                self.tasks = random.sample(self.training_tasks, train_num_task) # select k unique tasks
        else: 
            self.tasks = self.testing_tasks # during Meta-testing, all tasks should be in-order 

        for b, task in enumerate(self.tasks):  
            print('-'*20)
            print(f'task {b}: {task}')
            # 2.select k_support + k_query examples from task randomly
            if task == 'squad':
                dataset, label = self.load_and_cache_examples_squad(task, self.tokenizer, self.evaluate, self.evaluate_whole)
                if self.evaluate: train_dataset = self.load_and_cache_examples_squad(task, self.tokenizer, True, self.evaluate_whole)
            elif task in glue_processors.keys():
                dataset, label = self.load_and_cache_examples_glue(task, self.tokenizer, self.evaluate, self.evaluate_whole) # map style dataset 
                if self.evaluate: train_dataset = self.load_and_cache_examples_glue(task, self.tokenizer, True, self.evaluate_whole)
            else:
                dataset, label = self.load_and_cache_examples_superglue(task, self.tokenizer, self.evaluate, self.evaluate_whole) # map style dataset 
                if self.evaluate: train_dataset = self.load_and_cache_examples_superglue(task, self.tokenizer, True, self.evaluate_whole)
            self.labels += [label]
            if self.evaluate and self.evaluate_whole:  # evaluate support: entire training set, query: entire dev set
                exam_test = dataset # dev set 
                # in meta-testing, the support set is further sampled down to support size. therefore returning the entire set heree
                exam_train = train_dataset 
            else:
                exam_train, exam_test = random_split(dataset, [self.k_support, self.k_query])

            # 3. put into support and queries 
            self.supports.append(exam_train)
            self.queries.append(exam_test)

            
    ###################################
    #### dataloader of Super-GLUE  ####
    ###################################
    def load_and_cache_examples_superglue(self, task, tokenizer, evaluate=False, evaluate_whole_set = False):
        '''
        Heavily insipired from official loading and cache scripts from Huggingface Transformer func load_and_cache_examples
        https://github.com/huggingface/transformers/blob/master/examples/run_glue.py#L334
        NOTE: currently does not support any reading comprehension tasks, namely MultiRC and ReCoRD. 
        '''
        folder_name = {'boolq': 'BoolQ', 'multirc':'MultiRC', 'record': 'ReCoRD', 'wic': 'WiC', 'rte2': 'RTE'}
        if task in folder_name:
            task_data_path = folder_name[task]
        else:
            task_data_path = task.upper()


        if self.local_rank not in [-1, 0] and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        processor = superglue_processors[task]()
        output_mode = superglue_output_modes[task]
        cached_downloaded_file = os.path.join(self.data_dir, 'superglue', task_data_path)
        #print(cached_downloaded_file)
        cached_features_file = os.path.join(
        self.data_dir,
        'superglue',
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            str(self.bert_model),
            str(self.max_seq_length),
            str(task),
            ),
        )
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)

        else: 
            logger.info(f"Creating features from dataset file at {cached_downloaded_file}")
            label_list = processor.get_labels()

            examples = (processor.get_dev_examples(cached_downloaded_file) if evaluate 
                       else processor.get_train_examples(cached_downloaded_file))

            features = superglue_convert_examples_to_features(
                        examples, tokenizer, max_length=self.max_seq_length, 
                        task = task,label_list=label_list, output_mode=output_mode, model_type = self.bert_model)
            if self.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        if self.local_rank == 0 and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        if len(features) < self.k_query + self.k_support:
            selected_features = random.choices(features, k = self.k_support + self.k_query)
        else:
            selected_features = random.sample(features, self.k_support + self.k_query)

        if evaluate and evaluate_whole_set: # extracting entire dev set 
            selected_features = features

        # Convert to Tensors and build dataset
        if task == 'copa': 
            all_input_ids = list(map(lambda x: torch.tensor([f.input_ids for f in x], dtype = torch.long), selected_features))
            all_attention_mask = list(map(lambda x: torch.tensor([f.attention_mask for f in x], dtype=torch.long), selected_features))
            all_token_type_ids = list(map(lambda x: torch.tensor([f.token_type_ids for f in x], dtype=torch.long), selected_features))

            if output_mode == "classification":
                all_labels = list(map(lambda x: torch.tensor([f.label for f in x], dtype=torch.long), selected_features))
            elif output_mode == "regression":
                all_labels = list(map(lambda x: torch.tensor([f.label for f in x], dtype=torch.float), selected_features))

            dataset = ( TensorDataset(all_input_ids[0], all_attention_mask[0], all_token_type_ids[0], all_labels[0]), 
                        TensorDataset(all_input_ids[1], all_attention_mask[1], all_token_type_ids[1], all_labels[1]) )

        elif task in ['wic', 'wsc']:
            all_input_ids = torch.tensor([f.input_ids for f in selected_features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.input_mask for f in selected_features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.segment_ids for f in selected_features], dtype=torch.long)
            all_span_1_mask = torch.tensor([f.span_1_mask for f in selected_features], dtype=torch.long)
            all_span_1_text = torch.tensor([f.span_1_text for f in selected_features])
            all_span_2_mask = torch.tensor([f.span_2_mask for f in selected_features], dtype=torch.long)
            all_span_2_text = torch.tensor([f.span_2_text for f in selected_features]) 

            if output_mode == "classification":
                all_labels = torch.tensor([f.label for f in selected_features], dtype=torch.long)
            elif output_mode == "regression":
                all_labels = torch.tensor([f.label for f in selected_features], dtype=torch.float)

            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels,
                                    all_span_1_mask, all_span_1_text, all_span_2_mask, all_span_2_text
                                    )

        else: #for ['boolq', 'cb', 'rte']
            all_input_ids = torch.tensor([f.input_ids for f in selected_features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in selected_features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in selected_features], dtype=torch.long)
            if output_mode == "classification":
                all_labels = torch.tensor([f.label for f in selected_features], dtype=torch.long)
            elif output_mode == "regression":
                all_labels = torch.tensor([f.label for f in selected_features], dtype=torch.float)

            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

        return dataset, output_mode

    ### dataloader for SQuaD:
    def load_and_cache_examples_squad(self, tokenizer, evaluate=False, evaluate_whole_set = False):
        '''
        Heavily insipired from official loading and cache scripts from Huggingface Transformer func load_and_cache_examples
        https://github.com/huggingface/transformers/blob/master/examples/run_squad.py
        '''
        if self.local_rank not in [-1, 0] and not evaluate:
            # Make sure only the first process in distributed training process the dataset, and the others will use the cache
            torch.distributed.barrier()

        processor = SquadV2Processor() 
        
        cached_features_file = os.path.join(
        self.data_dir,
        'squad',
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            str(self.bert_model),
            str(self.max_seq_length),
            str(task),
            ),
        )
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            features_and_dataset= torch.load(cached_features_file)
            _, dataset = (
                features_and_dataset["features"],
                features_and_dataset["dataset"])
        else: 
            logger.info(f"Creating features from dataset file at {cached_downloaded_file}")
            if evaluate:
                examples = processor.get_dev_examples(self.data_dir+'/SQUAD', filename=processor.train_file)
            else:
                examples = processor.get_train_examples(self.data_dir+'/SQUAD', filename=processor.dev_file)

            features, dataset = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=self.max_seq_length,
                doc_stride=self.doc_stride,
                max_query_length=self.max_query_length,
                is_training=not evaluate,
                return_dataset="pt", # returns a torch.data.TensorDataset
                threads=1, # higher if multiple processing threads
              ) 

            if self.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save({"features": features, "dataset": dataset}, cached_features_file)

        if self.local_rank == 0 and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        if evaluate and evaluate_whole_set: # extracting entire dev set 
            return dataset 
        else:
            return_dataset, _ = random_split(dataset, [self.k_support + self.k_query, len(dataset)-(self.k_support + self.k_query)])
            return return_dataset, 'squad'

    # dataloader of GLUE 
    def load_and_cache_examples_glue(self, task, tokenizer, evaluate=False, evaluate_whole_set=False):
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

        processor = glue_processors[task]()
        output_mode = glue_output_modes[task]
        cached_downloaded_file = os.path.join(self.data_dir, 'glue_data', task_data_path)
        #print(cached_downloaded_file)

        cached_features_file = os.path.join(
        self.data_dir,
        'glue_data',
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            str(self.bert_model),
            str(self.max_seq_length),
            str(task),
            ),
        )
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)

        else: 
            logger.info(f"Creating features from dataset file at {cached_downloaded_file}")
            label_list = processor.get_labels()

            examples = (processor.get_dev_examples(cached_downloaded_file) if evaluate 
                       else processor.get_train_examples(cached_downloaded_file))

            features = glue_convert_examples_to_features(
                examples, tokenizer, max_length=self.max_seq_length, label_list=label_list, output_mode=output_mode,)

            if self.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        if self.local_rank == 0 and not evaluate:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        if len(features) < self.k_query + self.k_support:
            selected_features = random.choices(features, k = self.k_support + self.k_query)
        else:
            selected_features = random.sample(features, self.k_support + self.k_query)

        if evaluate and evaluate_whole_set: # extracting entire dev set 
            selected_features = features

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in selected_features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in selected_features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in selected_features], dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in selected_features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in selected_features], dtype=torch.float)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    
        return dataset, output_mode

    def __getitem__(self, index):
        support_set = self.supports[index]
        query_set   = self.queries[index]
        return support_set, query_set

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.num_task_train
    
    def get_tasks_and_modes(self):
        return [*zip(self.tasks, self.labels)]