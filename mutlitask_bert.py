'''
ref: https://github.com/IBM/superglue-mtl/blob/master/modeling/multitask_modeling.py
'''
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
#from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
from transformers import BertModel, BertPreTrainedModel

class SpanClassifier(nn.Module):
    """given the span embeddings, classify whether they have relations"""
    def __init__(self, d_inp):
        super(SpanClassifier, self).__init__()
        self.d_inp = d_inp
        self.bilinear_layer = nn.Bilinear(d_inp, d_inp, 1)
        self.output = nn.Sigmoid()
        self.loss = BCELoss()


    def forward(self, span_emb_1, span_emb_2, label=None):
        """Calculate the similarity as bilinear product of span embeddings.
        Args:
            span_emb_1: [batch_size, hidden] (Tensor) hidden states for span_1
            span_emb_2: [batch_size, hidden] (Tensor) hidden states for span_2
            label: [batch_size] 0/1 Tensor, if none is supplied do prediction.
        """
        similarity = self.bilinear_layer(span_emb_1, span_emb_2)
        probs = self.output(similarity)
        outputs = (similarity,)
        if label is not None:
            cur_loss = self.loss(probs, label)
            outputs = (cur_loss,) + outputs 
        return outputs


class BertForSpanClassification(BertPreTrainedModel):
    """For span classification tasks such as WiC or WSC."""
    def __init__(self, config, task_num_labels, tasks):
        super(BertForSpanClassification, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = SpanClassifier(d_inp=config.hidden_size)
        #self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, span_1, span_2, labels=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask, position_ids=None, head_mask=None)
        span1_emb = self._extract_span_emb(outputs[0], span_1)
        span2_emb = self._extract_span_emb(outputs[0], span_2)
        outputs = self.classifier(span1_emb, span2_emb, labels)
        return outputs

    def _extract_span_emb(self, sequence_outputs, span):
        """Extract embeddings for spans, sum up when span is multiple bpe units.
        Args:
            sequence_outputs: [batch_size x max_seq_length x hidden_size] (Tensor) The last layer hidden states for
                all tokens.
            span: list(str). The list of token ids corresponding to the span
        """
        prod = sequence_outputs * span.unsqueeze(-1).float()
        emb_sum = prod.sum(dim=-2)
        return emb_sum



class BertForSequenceClassificationMultiTask(BertPreTrainedModel):
    """
    BERT model for classification with mutiple linear layers for multi-task setup
     Params:
        `config`: a BertConfig class instance with the configuration to build a new model
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False
        `task_num_labels`: the number of classes for each classifier
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
        `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
            It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
    Outputs:
        if `labels` is not `None`: # for task such as 
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    """

    def __init__(self, config, task_num_labels, tasks):
        """
        Initiate BertForSequenceClassificationMultiTask with task informations.
         Params:
            `config`: a BertConfig class instance with the configuration to build a new model
            `task_num_labels`: a dictionary mapping task name to the number of labels for that task
            `tasks`: a list of task names. It has to be consistent with `task_num_labels`
        """
        super(BertForSequenceClassificationMultiTask, self).__init__(config)
        self.task_num_labels = task_num_labels
        self.bert = BertModel(config) # bare bert model output last hidden layers
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, task_num_labels[task]) for task in tasks])
        self.id2task = tasks
        self.task2id = {task: i for i, task in enumerate(tasks)}
        #self.apply(self.init_weights)

    def forward(self, task_id, input_ids, token_type_ids, attention_mask, labels=None):
        """ one batch can be only one task """
        outputs = self.bert(input_ids, token_type_ids, attention_mask, position_ids=None, head_mask=None)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        classifier = self.classifiers[task_id]
        logits = classifier(pooled_output)

        #outputs = (logits,) + outputs[2:]
        outputs = (logits, )
        num_labels = self.task_num_labels[self.id2task[task_id]]
        if labels is not None:
            if num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1)*5, labels.view(-1)) # multiply by 5 to match label interval [0, 5]
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1)) 
            outputs = (loss,) + outputs

        return outputs  # (loss), logits [W/O (hidden_states), (attentions)]
