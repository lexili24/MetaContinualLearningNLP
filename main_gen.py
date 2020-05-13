import json
from random import shuffle
from collections import Counter
import torch
from transformers import BertModel, BertTokenizer
import time
import logging
import argparse
import os
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#from oml import Learner
from multitask_both import Learner
from task_all import MetaTask
import random
import numpy as np


def random_seed(value):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)


def create_batch_of_tasks(ids, epoch, taskset,  task_list, is_shuffle=True, batch_size=4):
    idxs = list(range(0, len(taskset)))
    if is_shuffle:
        random.shuffle(idxs)
    batch_task = np.random.choice(idxs, batch_size, replace=False)
    for i in range(0, len(batch_task)):
        ids.append(task_list[idxs[batch_task[i]]])
    yield [taskset[batch_task[i]] for i in range(0, len(batch_task))]

    # for j in range(0,len(idxs), batch_size):
    #     ids[j] = [idxs[i] for i in range(j, min(j + batch_size,len(taskset)))]
    #     print('ids:',ids)
    #     yield [taskset[idxs[i]] for i in range(j, min(j + batch_size,len(taskset)))]


def create_test_tasks(idt, epoch, taskset, is_shuffle=False, batch_size=3):
    idxs = list(range(0, batch_size))
    for i in range(0, batch_size):
        idt.append(idxs[i])
    yield [taskset[i] for i in range(0, batch_size)]

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Path to bert model")


    parser.add_argument("--epoch", default=150, type=int,
                        help="Number of outer interation")

    parser.add_argument("--k_spt", default=80, type=int,
                        help="Number of support samples per task")

    parser.add_argument("--k_qry", default=20, type=int,
                        help="Number of query samples per task")

    parser.add_argument("--outer_batch_size", default=4, type=int,
                        help="Batch of task size")

    parser.add_argument("--inner_batch_size", default=12, type=int,
                        help="Training batch size in inner iteration")

    parser.add_argument("--outer_update_lr", default=5e-5, type=float,
                        help="Meta learning rate")

    parser.add_argument("--inner_update_lr", default=5e-5, type=float,
                        help="Inner update learning rate")

    parser.add_argument("--inner_update_step", default=5, type=int,
                        help="Number of interation in the inner loop during train time")

    parser.add_argument("--inner_update_step_eval", default=7, type=int,
                        help="Number of interation in the inner loop during test time")

    parser.add_argument("--num_task_train", default=4, type=int,
                        help="Total number of meta tasks for training")

    parser.add_argument("--num_task_test", default=4, type=int,
                        help="Total number of tasks for testing")

    parser.add_argument("--data_dir", type=str,
                        help='parent directory with following folders: glue_data, superglue and squad')

    parser.add_argument("--max_seq_length", default=128, type=int,
                        help='max length of a input sequence the model could take')

    parser.add_argument("--local_rank", default=-1, type=int,
                        help='parallel computing')

    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")

    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                        "be truncated to this length.")

    parser.add_argument("--training_tasks", default=['copa','copa','copa','copa'], type=list,
                        #choices = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'snli', 'sst-2', 'sts-b', 'wnli']
                        help="Define meta-training tasks list.") # ['sst-2', 'mnli', 'qqp', 'qnli']

    parser.add_argument("--testing_tasks", default=['cola', 'mrpc', 'sts-b', 'rte'], type=list,
                        help="Define meta-testing tasks list.")

    parser.add_argument("--evaluate_whole_set", default=True, type=bool,
                        help="Indicator on whether evaluate entire dev set during meta-testing phase")

    parser.add_argument("--meta_testing_size", default=100, type=int,
            help="Specifiy number of training samples to draw to feed into meta-testing.")

    parser.add_argument("--oml", default=False, type=bool,
            help="Turn on this flag to run on oml")

    args = parser.parse_args()

    ###  log 
    if args.oml == True:
        print('='*35, 'Training OML', '='*35)
    else:
        print('='*35, 'Training MAML', '='*35)
    print('meta training tasks', args.training_tasks)
    print('meta testing tasks', args.testing_tasks)

    ## initialize model
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    learner = Learner(args)
    print('start reading test data', flush=True)
    test = MetaTask(args=args, num_task=args.num_task_test, k_support=args.k_spt,
                    k_query=args.k_qry, tokenizer=tokenizer, max_seq_length=args.max_seq_length, evaluate=True)
    print('finish reading test data', flush=True)
    print('meta testing training samples', args.meta_testing_size)
    global_step = 0
    for epoch in range(args.epoch):
        ids = []
        train = MetaTask(args=args, num_task=args.num_task_train, k_support=args.k_spt,
                         k_query=args.k_qry, tokenizer=tokenizer, max_seq_length=args.max_seq_length, evaluate=False)
        db = create_batch_of_tasks(ids, epoch, train, is_shuffle=True, batch_size=args.outer_batch_size, task_list = args.training_tasks)
        task_batch = next(db)
        print("\n-----------------Meta-Training Mode-----------------\n", flush=True)
        # print('ids', ids)
        # print('task_batch size', len(task_batch))
        acc = learner(ids, task_batch)
        print('Step:', epoch, '\tAvg Acc in query set:', acc)
        del train
        del db

        if epoch % 5 == 0:
            random_seed(123)
            print("\n-----------------Meta-Testing Mode-----------------\n")
            idt = []
            db_test = create_test_tasks(idt, epoch, test, is_shuffle=False, batch_size=3, task_list = args.testing_tasks)
            test_batch = next(db_test)
            acc = learner.finetune(idt, test_batch)

            print('Step:', epoch, 'Test F1:', np.mean(acc))
            print('\n')
            random_seed(int(time.time() % 10))
            del db_test


if __name__ == "__main__":
    main()
