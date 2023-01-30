import random
from typing import List, Tuple
import datasets
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding

from tevatron.arguments import DataArguments
from tevatron.trainer import TevatronTrainer

class BITrainPreProcessor:
    def __init__(self, tokenizer, query_max_length=32, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        query = self.tokenizer.encode(example['query'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        positives_source = []
        for pos in example['positive_passages_source']:
            text = pos['title'] + self.separator + pos['text'] if 'title' in pos else pos['text']
            positives_source.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
        negatives_source = []
        for neg in example['negative_passages_source']:
            text = neg['title'] + self.separator + neg['text'] if 'title' in neg else neg['text']
            negatives_source.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
        positives_target = []
        for pos in example['positive_passages_target']:
            text = pos['title'] + self.separator + pos['text'] if 'title' in pos else pos['text']
            positives_target.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
        negatives_target = []
        for neg in example['negative_passages_target']:
            text = neg['title'] + self.separator + neg['text'] if 'title' in neg else neg['text']
            negatives_target.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
        return {'query': query, 'positives_source': positives_source, 'negatives_source': negatives_target, 'positives_target': positives_target, 'negatives_target': negatives_target}

class HFBITrainDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        data_files = data_args.train_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, cache_dir=cache_dir, use_auth_token=True)[data_args.dataset_split]
        self.preprocessor = BITrainPreProcessor
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.neg_num = data_args.train_n_passages - 1
        self.separator = getattr(self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator)

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len, self.p_max_len, self.separator),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenizer on train dataset",
            )
        return self.dataset

class BITrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: TevatronTrainer = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)

    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.prepare_for_model(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group['query']
        encoded_query = self.create_one_example(qry, is_query=True)

        encoded_passages_source = []
        encoded_passages_target = []
        group_positives_source = group['positives_source']
        group_negatives_source = group['negatives_source']
        group_positives_target = group['positives_target']
        group_negatives_target = group['negatives_target']

        if self.data_args.positive_passage_no_shuffle:
            pos_psg_source = group_positives_source[0]
            pos_psg_target = group_positives_source[0]
        else:
            pos_psg_source = group_positives_source[(_hashed_seed + epoch) % len(group_positives_source)]
            pos_psg_target = group_positives_target[(_hashed_seed + epoch) % len(group_positives_source)]
        encoded_passages_source.append(self.create_one_example(pos_psg_source))
        encoded_passages_target.append(self.create_one_example(pos_psg_target))

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives_source) < negative_size:
            negs_idx = random.choices(range(len(group_negatives_source)), k=negative_size)
            negs_source = [group_negatives_source[i] for i in negs_idx]
            negs_target = [group_negatives_target[i] for i in negs_idx]
        elif self.data_args.train_n_passages == 1:
            negs_source = []
            negs_target = []
        elif self.data_args.negative_passage_no_shuffle:
            negs_source = group_negatives_source[:negative_size]
            negs_target = group_negatives_target[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives_source)
            negs_source = [x for x in group_negatives_source]
            negs_target = [x for x in group_negatives_target]
            random.Random(_hashed_seed).shuffle(negs_source)
            random.Random(_hashed_seed).shuffle(negs_target)
            negs_source = negs_source * 2
            negs_target = negs_target * 2
            negs_source = negs_source[_offset: _offset + negative_size]
            negs_target = negs_target[_offset: _offset + negative_size]

        for neg_psg in negs_source:
            encoded_passages_source.append(self.create_one_example(neg_psg))
        for neg_psg in negs_target:
            encoded_passages_target.append(self.create_one_example(neg_psg))
        return encoded_query, encoded_passages_source + encoded_passages_target
