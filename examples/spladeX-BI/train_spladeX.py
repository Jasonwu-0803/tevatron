import logging
import os
import sys
import json

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from dataclasses import dataclass, field
from tevatron.arguments import ModelArguments, DataArguments, TevatronTrainingArguments
from tevatron.data import TrainDataset, QPCollator
from tevatron.modeling import SpladeModel, SpladeXModel
from tevatron.trainer import TevatronTrainer
from tevatron.datasets import HFTrainDataset
from tevatron.BITrain import HFBITrainDataset, BITrainDataset

logger = logging.getLogger(__name__)


@dataclass
class SpladeTrainingArguments(TevatronTrainingArguments):
    q_flops_loss_factor: float = field(default=4)
    p_flops_loss_factor: float = field(default=32)


class SpladeXTrainer(TevatronTrainer):
    def __init__(self, *args, **kwargs):
        super(SpladeXTrainer, self).__init__(*args, **kwargs)
        self.world_size = dist.get_world_size() if self.args.negatives_x_device else 1

    @staticmethod
    def _flops(inputs):
        return torch.sum(torch.mean(torch.abs(inputs), dim=0) ** 2)

    def compute_loss(self, model, inputs):
        query, passage = inputs
        passage_source = {}
        passage_target = {}
        epc_len = len(query["input_ids"])
        psg_len = len(passage['input_ids'])
        train_n_psg = int(psg_len / epc_len)
        passage_source['input_ids'] = [passage['input_ids'][i] for i in range(psg_len) if i % (2*train_n_psg) < train_n_psg]
        passage_source['attention_mask'] = [passage['attention_mask'][i] for i in range(psg_len) if i % (2*train_n_psg) < train_n_psg]
        passage_target['input_ids'] = [passage['input_ids'][i] for i in range(psg_len) if i % (2*train_n_psg) >= train_n_psg]
        passage_target['attention_mask'] = [passage['attention_mask'][i] for i in range(psg_len) if i % (2*train_n_psg) >= train_n_psg]
        passage_source['input_ids'] = torch.stack(passage_source['input_ids'])
        passage_source['attention_mask'] = torch.stack(passage_source['attention_mask'])
        passage_target['input_ids'] = torch.stack(passage_target['input_ids'])
        passage_target['attention_mask'] = torch.stack(passage_target['attention_mask'])
        
        output_source = model(query=query, passage=passage_source)
        q_reps_source = output_source.q_reps
        p_reps_source = output_source.p_reps
        loss_source = output_source.loss
        q_flops_loss_source = self.args.q_flops_loss_factor*self._flops(q_reps_source)
        p_flops_loss_source = self.args.p_flops_loss_factor*self._flops(p_reps_source)
        
        output_target = model(query=query, passage=passage_target)
        q_reps_target = output_target.q_reps
        p_reps_target = output_target.p_reps
        loss_target = output_target.loss
        q_flops_loss_target = self.args.q_flops_loss_factor*self._flops(q_reps_target)
        p_flops_loss_target = self.args.p_flops_loss_factor*self._flops(p_reps_target)
        
        if self.args.negatives_x_device:
            q_flops_loss_source *= self.world_size
            p_flops_loss_source *= self.world_size

            q_flops_loss_target *= self.world_size
            p_flops_loss_target *= self.world_size
        MSE = torch.nn.MSELoss()
        return loss_source + q_flops_loss_source + p_flops_loss_source + loss_target + q_flops_loss_target + p_flops_loss_target + MSE(p_reps_source, p_reps_target)


TrainingArguments = SpladeTrainingArguments

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)
    #logger.info("data parameters %s", data_args)
    #print(data_args.dataset_language)
    #print(data_args.dataset_split)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = SpladeXModel.build(
        model_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    #fetch source laguage mask
    mask = {}
    with open('mask.txt', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            mask = data
    model.mask = mask

    train_dataset = HFBITrainDataset(tokenizer=tokenizer, data_args=data_args,
                                   cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    train_dataset = BITrainDataset(data_args, train_dataset.process(), tokenizer)

    trainer = SpladeXTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=QPCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
    )
    train_dataset.trainer = trainer

    trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
