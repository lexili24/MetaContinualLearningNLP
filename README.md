# Meta Learning under contiual learning framework 
## Requirements
  - transformers==2.2.1
  - python>=3.6
  - torch==1.3.0

# Introduction
This repository is an implementation of First-order MAML under continual learning on BERT training on GLUE tasks. The work is divided into two fold:
## Contiual MAML: MAML is adaptaed from [meta learning bert](https://github.com/mailong25/meta-learning-bert), and modified to support contiual learning
## Dataloader: dataloader is adapted and modified from [meta learning bert](https://github.com/mailong25/meta-learning-bert) and [transfomers](https://github.com/huggingface/transformers) to provide batch of GLUE tasks. 
Currently, there are two implementation of dataloader in `task_glue` and `task_glue_wo_saving`. Both scripts have the same inputs and outputs, but have different processing times. Specifically, `task_glue` preprocesses texts and saved in local, while `task_glue_wo_saving` processes features from text online without saving. Therefore, `task_glue` might have a slower time when a dataset is called first-time. 
