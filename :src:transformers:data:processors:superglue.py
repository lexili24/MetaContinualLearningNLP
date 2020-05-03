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
""" super-GLUE processors and helpers """

import logging
import os
import re
from ...file_utils import is_tf_available
from .utils import DataProcessor, InputExample, InputFeatures, InputFeatures_w, COPAInputExample, WSCInputExample, WiCInputExample


if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def superglue_convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
    model_type = 'bert-base-uncased'
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples``, ``COPAInputExamples``,  ``WiCInputExamples``, ``WSCInputExamples``
                  or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = superglue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = superglue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def featurize_example_standard(examples = examples, processor = processor, tokenizer = tokenizer,
        mask_padding_with_zero = mask_padding_with_zero):
        features = []
        for (ex_index, example) in enumerate(examples):
            len_examples = 0
            if is_tf_dataset:
                example = processor.get_example_from_tensor_dict(example)
                example = processor.tfds_map(example)
                len_examples = tf.data.experimental.cardinality(examples)
            else:
                len_examples = len(examples)
            if ex_index % 10000 == 0:
                logger.info("Writing example %d/%d" % (ex_index, len_examples))

            inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length)
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
                    len(attention_mask), max_length)
            assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
                    len(token_type_ids), max_length)

            if output_mode == "classification":
                label = label_map[example.label]
            elif output_mode == "regression":
                label = float(example.label)
            else:
                raise KeyError(output_mode)

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("label: %s (id = %d)" % (example.label, label))

            features.append(
                InputFeatures(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    token_type_ids=token_type_ids, 
                    label=label
                )
            )

        if is_tf_available() and is_tf_dataset:

            def gen():
                for ex in features:
                    yield (
                        {
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "token_type_ids": ex.token_type_ids,
                        },
                        ex.label,
                    )

            return tf.data.Dataset.from_generator(
                gen,
                ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
                (
                    {
                        "input_ids": tf.TensorShape([None]),
                        "attention_mask": tf.TensorShape([None]),
                        "token_type_ids": tf.TensorShape([None]),
                    },
                    tf.TensorShape([]),
                ),
            )


        return features

    def featurize_example_copa(examples = examples, processor = processor, tokenizer = tokenizer,
        mask_padding_with_zero = mask_padding_with_zero):

        def _featurize_example(text_a, text_b, text_c, guid, cur_label=None, print_example=False,
                                max_length = max_length, model_type = model_type,
                                mask_padding_with_zero = mask_padding_with_zero):
            tokens_a = tokenizer.tokenize(text_a)
            tokens_b = tokenizer.tokenize(text_b)
            tokens_c = tokenizer.tokenize(text_c)
            special_tokens_count = 6 if "roberta" in model_type else 4
            _truncate_seq_pair(tokens_a, tokens_c, max_length - special_tokens_count - len(tokens_b))
            tokens = tokens_a + [tokenizer.sep_token]
            if "roberta" in model_type:
                tokens += [tokenizer.sep_token]
            segment_ids = [0] * len(tokens)

            tokens += tokens_b + [tokenizer.sep_token]
            segment_ids += [1] * (len(tokens_b) + 1)
            if "roberta" in model_type:
                tokens += [tokenizer.sep_token]
                segment_ids += [1]

            tokens += tokens_c + [tokenizer.sep_token]
            segment_ids += [2] * (len(tokens_c) + 1)

            tokens = [tokenizer.cls_token] + tokens
            segment_ids = [0] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = tokenizer.convert_tokens_to_ids([tokenizer.pad_token] * padding_length) + input_ids 
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([0] * padding_length) + segment_ids
            else:
                input_ids = input_ids + tokenizer.convert_tokens_to_ids([tokenizer.pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([0] * padding_length)

            label_id = float(cur_label) if cur_label is not None else None

            assert len(input_ids) == max_length
            assert len(input_mask) == max_length
            assert len(segment_ids) == max_length

            if print_example:
                logging.info("*** Example (COPA) ***")
                logging.info("guid: %s" % (guid))
                logging.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logging.info("label: %s (id = %s)" % (str(cur_label), str(label_id)))

            return InputFeatures(input_ids=input_ids,
                                 attention_mask=input_mask,
                                 token_type_ids=segment_ids,
                                 label=label_id)
            
        features_1 = []
        features_2 = []
        for (ex_index, example) in enumerate(examples):
            len_examples = 0
            if is_tf_dataset:
                example = processor.get_example_from_tensor_dict(example)
                example = processor.tfds_map(example)
                len_examples = tf.data.experimental.cardinality(examples)
            else:
                len_examples = len(examples)
            if ex_index % 10000 == 0:
                logger.info("Writing example %d/%d" % (ex_index, len_examples))

            features_1.append(_featurize_example(example.text_a,
                                        example.question,
                                        example.text_pre,
                                        cur_label=int(example.label == '0'),
                                        print_example=True,
                                        guid = example.guid))
            features_2.append(_featurize_example(example.text_b,
                                        example.question,
                                        example.text_pre,
                                        cur_label=int(example.label == '1'),
                                        print_example=True,
                                        guid = example.guid))
        return features_1, features_2

    def featurize_example_wsc(examples = examples, processor = processor, tokenizer = tokenizer, mask_padding_with_zero = mask_padding_with_zero):

        def _featurize_example(example, max_seq_length, tokenizer = tokenizer, 
                    label_map=label_map, model_type=model_type, print_example=False, mask_padding_with_zero = mask_padding_with_zero):
            """Tokenize example for WSC.
            Args:
                tokenizer: either a BertTokenizer or a RobertaTokenizer
                max_seq_length: int. The maximum allowed number of bpe units for the input.
                label_map: dictionary. A map that returns the label_id given the label string.
                model_type: string. Either `bert` or `roberta`. For `roberta` there will be an extra sep token in
                        the middle.
                print_example: bool. If set to True, print the tokenization information for current instance.
            """
            tokens_a = tokenizer.tokenize(example.text)
            token_word_ids = _get_word_ids(tokens_a, model_type)
            span_1_tok_ids = _get_token_ids(token_word_ids, example.span_1[0], offset=1)
            span_2_tok_ids = _get_token_ids(token_word_ids, example.span_2[0], offset=1)

            special_tokens_count = 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:max_seq_length - special_tokens_count]

            tokens = tokens_a + [tokenizer.sep_token]
            segment_ids = [0] * len(tokens)

            tokens = [tokenizer.cls_token] + tokens
            segment_ids = [0] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids) 
            
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = tokenizer.convert_tokens_to_ids([tokenizer.pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = [0] * padding_length + segment_ids
            else:
                input_ids = input_ids + tokenizer.convert_tokens_to_ids([tokenizer.pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + [0] * padding_length


            span_1_mask = [0] * len(input_ids)
            for k in span_1_tok_ids:
                if pad_on_left:
                    span_1_mask[k+padding_length] = 1
                else:
                    span_1_mask[k] = 1

            span_2_mask = [0] * len(input_ids)
            for k in span_2_tok_ids:
                if pad_on_left:
                    span_2_mask[k+padding_length] = 1
                else: 
                    span_2_mask[k] = 1

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(span_1_mask) == max_seq_length
            assert len(span_2_mask) == max_seq_length


            if output_mode == "classification":
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)


            if print_example:
                logging.info("*** Example (%s) ***" % task)
                logging.info("guid: %s" % (example.guid))
                logging.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logging.info("label: %s (id = %s)" % (str(example.label), str(label_id)))

            return InputFeatures_w(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                span_1_mask=span_1_mask,
                                span_1_text=tokenizer.convert_tokens_to_ids(example.span_1[1]),
                                span_2_mask=span_2_mask,
                                span_2_text=tokenizer.convert_tokens_to_ids(example.span_2[1]),
                                label=label_id)
            
        
        features = []
        for (ex_index, example) in enumerate(examples):
            len_examples = 0
            if is_tf_dataset:
                example = processor.get_example_from_tensor_dict(example)
                example = processor.tfds_map(example)
                len_examples = tf.data.experimental.cardinality(examples)
            else:
                len_examples = len(examples)
            if ex_index % 10000 == 0:
                logger.info("Writing example %d/%d" % (ex_index, len_examples))

            features.append(_featurize_example(example, tokenizer = tokenizer, max_seq_length= max_length, 
                                                label_map=label_map, model_type=model_type, print_example=True))

        return features

    def featurize_example_wic(examples = examples, processor = processor, tokenizer = tokenizer, mask_padding_with_zero = mask_padding_with_zero):

        def _featurize_example(example, max_seq_length, tokenizer = tokenizer , 
                    label_map=label_map, model_type=model_type, print_example=False, mask_padding_with_zero = mask_padding_with_zero):
            """Tokenize example for WiC.
            Args:
                tokenizer: either a BertTokenizer or a RobertaTokenizer
                max_seq_length: int. The maximum allowed number of bpe units for the input.
                label_map: dictionary. A map that returns the label_id given the label string.
                model_type: string. Either `bert` or `roberta`. For `roberta` there will be an extra sep token in
                        the middle.
                print_example: bool. If set to True, print the tokenization information for current instance.
            """
            
            tokens_a = tokenizer.tokenize(example.sent1)
            index1, word1 = _digits_to_index(tokenizer, example.sent1, example.idxs1)
            token_word_ids_a = _get_word_ids(tokens_a, model_type)
            sent_1_tok_ids = _get_token_ids(token_word_ids_a, index1, offset=1)

            tokens_b = tokenizer.tokenize(example.sent2)
            index2, word2 = _digits_to_index(tokenizer, example.sent2, example.idxs2)
            token_word_ids_b = _get_word_ids(tokens_b, model_type)
            sent_2_tok_ids = _get_token_ids(token_word_ids_b, index2, offset=1)

            special_tokens_count = 5 if "roberta" in model_type else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
            tokens = tokens_a + [tokenizer.sep_token] # +1 
            if "roberta" in model_type:
                tokens += [tokenizer.sep_token] # (+1)
            segment_ids = [0] * len(tokens)
            sent1_len = len(tokens)
            tokens += tokens_b + [tokenizer.sep_token] # +1 
            segment_ids += [1] * (len(tokens_b) + 1)
            if "roberta" in model_type:
                tokens += [tokenizer.sep_token] # (+1)
                segment_ids += [1]

            tokens = [tokenizer.cls_token] + tokens # +1 
            segment_ids = [0] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = tokenizer.convert_tokens_to_ids([tokenizer.pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = [0] * padding_length + segment_ids
            else:
                input_ids = input_ids + tokenizer.convert_tokens_to_ids([tokenizer.pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + [0] * padding_length


            span_1_mask = [0] * len(input_ids)
            for k in sent_1_tok_ids:
                if pad_on_left:
                    span_1_mask[k+padding_length] = 1
                else:
                    span_1_mask[k] = 1

            span_2_mask = [0] * len(input_ids)
            for k in sent_2_tok_ids:
                if pad_on_left:
                    span_2_mask[k + padding_length + sent1_len] = 1
                else: 
                    span_2_mask[k + sent1_len] = 1

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(span_1_mask) == max_seq_length
            assert len(span_2_mask) == max_seq_length


            if output_mode == "classification":
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)


            if print_example:
                logging.info("*** Example (%s) ***" % task)
                logging.info("guid: %s" % (example.guid))
                logging.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logging.info("label: %s (id = %s)" % (str(example.label), str(label_id)))

            return InputFeatures_w(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                span_1_mask=span_1_mask,
                                span_1_text=tokenizer.convert_tokens_to_ids(word1),
                                span_2_mask=span_2_mask,
                                span_2_text=tokenizer.convert_tokens_to_ids(word2),
                                label=label_id)
            
        
        features = []
        for (ex_index, example) in enumerate(examples):
            len_examples = 0
            if is_tf_dataset:
                example = processor.get_example_from_tensor_dict(example)
                example = processor.tfds_map(example)
                len_examples = tf.data.experimental.cardinality(examples)
            else:
                len_examples = len(examples)
            if ex_index % 10000 == 0:
                logger.info("Writing example %d/%d" % (ex_index, len_examples))

            features.append(_featurize_example(example, tokenizer = tokenizer, max_seq_length=max_length, 
                                                label_map=label_map, model_type=model_type, print_example=True))

        return features

        
    if task in ['boolq', 'rte2', 'cb']:
        return featurize_example_standard(examples, processor, tokenizer)
    elif task == 'copa': 
        return featurize_example_copa(examples, processor, tokenizer) # return a tuple 
    elif task == 'wsc': 
        return featurize_example_wsc(examples, processor, tokenizer)
    elif task == 'wic':
        return featurize_example_wic(examples, processor, tokenizer)

######### HELPER FUNCS #########
# credit: https://github.com/IBM/superglue-mtl/blob/1eb3e581c0ef3b4c261e0256ec26116d2b657c40/data_utils.py#L720

#### Helper func for COPA  #####
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

#### helper func for wic #####
# Given the beginning and the ending digit of the word in a sentenece, return 
# the index of the word in the sentence and the corresponding word.
def _digits_to_index(tokenizer, sent, idxs): 
    start, end = idxs
    word = sent[start:end]
    words = re.split(r'\W+', sent)
    #words = tokenizer.tokenize(sent)
    try:
        index = words.index(word)
    except:
        print('word', word)
        print('sent', words)
    return index, word 


#### Helper funcs for WSC ##### 
# given the original span of the word in original string, 
# these two funcs are constructed so that the span of target word
# in encoded string is returned. 
def _get_token_ids(token_word_ids, span_word_id, offset=1):
    """Retrieve token ids based on word ids.
    Args:
        token_word_ids: the list of word ids for token.
        span_word_id: int. the word id in the original string.
        offset: int. if the tokenized sequence is prepended with special token, this offset will be set to
        the number of special tokens (for example, if [CLS] is added, then offset=1).
    For example, the token word ids can be:
     ['ir', 'an', 'Ġand', 'Ġaf', 'ghan', 'istan', 'Ġspeak', 'Ġthe', 'Ġsame', 'Ġlanguage', 'Ġ.']
    And the original sentence is "iran and afghanistan speak the same language ."
    Suppose the span_word_id is 2 (afghanistan), then the token id is [3, 4, 5]
    """
    results = []
    for ix, word_id in enumerate(token_word_ids):
        if word_id == span_word_id:
            results.append(ix + offset)
        elif word_id > span_word_id:
            break
    return results


def _get_word_ids(tokens, model_type="bert"):
    """Given the BPE split results, mark each token with its original word ids.
    Args:
          tokens: a list of BPE units
    For example, if original sentnece is `iran and afghanistan speak the same language .`, then the roberta
    tokens will be:
    ['ir', 'an', 'Ġand', 'Ġaf', 'ghan', 'istan', 'Ġspeak', 'Ġthe', 'Ġsame', 'Ġlanguage', 'Ġ.']
    The word ids will be:
    [0,     0,     1,     2,    2,      2,        3,        4,      5,      6,      7]
    Note: this method assume the original sentence is split by one space and is already tokenized.
    """

    word_ids = []
    for tok in tokens:
        if len(word_ids) == 0:
            word_ids.append(0)
            continue

        if "roberta" in model_type:
            if tok[0] != "Ġ":
                word_ids.append(word_ids[-1])
            else:
                word_ids.append(word_ids[-1] + 1)
        else:
            if tok[:1] == "##":
                word_ids.append(word_ids[-1])
            else:
                word_ids.append(word_ids[-1] + 1)
    return word_ids


####################################
############ SuperGlue #############
####################################

class BoolQProcessor(DataProcessor):
    """Processor for the BoolQ data set (Super-GLUE version)."""
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["passage"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_to_list(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_to_list(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return ['True', 'False']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[2])
            text_a = line[0]
            text_b = line[1]
            label = str(line[-1])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class CbProcessor(DataProcessor):
    """Processor for the CB data set (Super-GLUE version)."""
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            tensor_dict["label"].numpy(),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_to_list(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_to_list(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[-1])
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class RteProcessor_superglue(DataProcessor):
    """Processor for the RTE data set (Super-GLUE version). Similar to Glue task rte, but merged with more data"""
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            tensor_dict["label"].numpy(),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_to_list(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_to_list(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[-1])
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class CopaProcessor(DataProcessor):
    """Processor for the COPA data set (Super-GLUE version).
       Bert model needs to be modified to take a premise with a question to select if choice 1 or choice 2 is favored. 
    """
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return COPAInputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["choice1"].numpy().decode("utf-8"),
            tensor_dict["choice2"].numpy().decode("utf-8"),
            tensor_dict["question"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_to_list(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_to_list(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[-1])
            text_pre = line[0]
            text_a = line[1]
            text_b = line[2]
            question = "What was the cause of this?" if line[3] == 'cause' else "What happened as a result?"
            label = str(line[4])
            examples.append(COPAInputExample(
                guid=guid, text_pre = text_pre,
                text_a=text_a, text_b=text_b, 
                question = question, label=label))
        return examples

class WscProcessor(DataProcessor):
    """Processor for the Multi-RC data set (Super-GLUE version)."""
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return WSCInputExample(
            tensor_dict["target"]["idx"].numpy(),
            tensor_dict["text"].numpy().decode("utf-8"),
            (tensor_dict["target"]["span1_index"].numpy(), tensor_dict["target"]["span1_text"].numpy().decode("utf-8")),
            (tensor_dict["target"]["span2_index"].numpy(), tensor_dict["target"]["span2_text"].numpy().decode("utf-8")),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_to_dict(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_to_dict(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return ["True", "False"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for line in lines: # inputs now are dict 
            guid = "%s-%s" % (set_type, line['idx'])
            text = line['text']
            label = str(line['label'])
            span_1 = (line['target']['span1_index'], line['target']['span1_text'])
            span_2 = (line['target']['span2_index'], line['target']['span2_text'])
            examples.append(WSCInputExample(
                guid=guid, text = text,
                span_1=span_1, span_2=span_2, 
                label=label))
        return examples



class WicProcessor(DataProcessor):
    """Processor for the Multi-RC data set (Super-GLUE version)."""
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return WSCInputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            (tensor_dict['start1'].numpy(), tensor_dict["end1"].numpy()),
            (tensor_dict['start2'].numpy(), tensor_dict["end2"].numpy()),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_to_list(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json_to_list(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return ["True", "False"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for line in lines: # inputs now are dict 
            guid = "%s-%s" % (set_type, line[3])
            sent1 = line[1]
            sent2 = line[2]
            label = str(line[4])
            idxs1 = (line[5], line[7])
            idxs2 = (line[6], line[8])
            examples.append(WiCInputExample(
                guid = guid,
                sent1 = sent1, sent2 = sent2,
                idxs1 = idxs1, idxs2 = idxs2,
                label = label))
        return examples


superglue_tasks_num_labels = {
    "boolq": 2,
    "cb"   : 3,
    "rte2" : 2,
    "copa" : 2,
    "wsc"  : 2,
    "wic"  : 2,
}

superglue_processors = {
    "boolq": BoolQProcessor,
    "cb"   : CbProcessor,
    "rte2" : RteProcessor_superglue,
    "copa" : CopaProcessor,
    "wsc"  : WscProcessor,
    "wic"  : WicProcessor,
}

superglue_output_modes = {
    "boolq": 'classification',
    "cb"   : 'classification',
    "rte2" : 'classification',
    "copa" : 'classification',
    "wsc"  : 'classification',
    "wic"  : 'classification',
}
