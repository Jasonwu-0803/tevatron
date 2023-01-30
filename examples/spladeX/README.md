## Generate the token mask of the source language (English By Default)
```bash
python gen_mask.py
```

## Train SPLADEX ZS on MS MARCO
```bash
CUDA_VISIBLE_DEVICES=0 python train_spladeX.py \
  --output_dir model_msmarco_spladeX \
  --model_name_or_path distilbert-base-multilingual-cased \
  --save_steps 20000 \
  --dataset_name Tevatron/msmarco-passage \
  --fp16 \
  --per_device_train_batch_size 8 \
  --train_n_passages 8 \
  --learning_rate 5e-6 \
  --q_max_len 128 \
  --p_max_len 128 \
  --q_flops_loss_factor 0.01 \
  --p_flops_loss_factor 0.01 \
  --num_train_epochs 3 \
  --logging_steps 500 \
  --overwrite_output_dir
```

## Train SPLADEX TT on MS MARCO
```bash
CUDA_VISIBLE_DEVICES=0 python train_spladeX.py \
  --output_dir model_msmarco_spladeX \
  --model_name_or_path distilbert-base-multilingual-cased \
  --save_steps 20000 \
  --dataset_name JAWCF/mmarco-tt:spanish \
  --fp16 \
  --per_device_train_batch_size 8 \
  --train_n_passages 8 \
  --learning_rate 5e-6 \
  --q_max_len 128 \
  --p_max_len 128 \
  --q_flops_loss_factor 0.01 \
  --p_flops_loss_factor 0.01 \
  --num_train_epochs 3 \
  --logging_steps 500 \
  --overwrite_output_dir
```

## Encode SPLADE 

Take spanish as target language (english is source language by default), SPLADEX encoding can be done as follows:

```bash
mkdir -p encoding_splade/corpus
mkdir -p encoding_splade/query
for i in $(seq -f "%02g" 8 9)
do
python encode_splade.py \
  --output_dir encoding_spladeX \
  --model_name_or_path model_msmarco_spladeX \
  --tokenizer_name distilbert-base-multilingual-cased \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --dataset_name crystina-z/mmarco-corpus:spanish\
  --encoded_save_path encoding_spladeX/corpus-spanish/split${i}.jsonl\
  --encode_num_shard 10 \
  --encode_shard_index ${i}
done


python -m encode_splade \
  --output_dir encoding_spladeX \
  --model_name_or_path model_msmarco_spladeX \
  --tokenizer_name distilbert-base-multilingual-cased \
  --fp16 \
  --q_max_len 128 \
  --encode_is_qry \
  --per_device_eval_batch_size 128 \
  --dataset_name Tevatron/msmarco-passage/dev \
  --encoded_save_path encoding_spladeX/query/dev.tsv 
```

## Index SPLADE with anserini
In the following, we consider that [ANSERINI](https://github.com/castorini/anserini) is installed, with all its tools, in $PATH_ANSERINI
```
sh $PATH_ANSERINI/target/appassembler/bin/IndexCollection -collection JsonVectorCollection \
 -input encoding_spladeX/corpus-spanish \
 -index spladeX_anserini_index_spanish \
 -generator DefaultLuceneDocumentGenerator -impact -pretokenized \
 -threads 16

```

## Retrieve SPLADE with anserini

```
sh $PATH_ANSERINI/target/appassembler/bin/SearchCollection -hits 1000 -parallelism 32 \
 -index spladeX_anserini_index_spanish \
 -topicreader TsvInt -topics encoding_spladeX/query/dev.tsv\
 -output spladeX_result.trec -format trec \
 -impact -pretokenized
```

## Evaluate SPLADE with anserini

```
$PATH_ANSERINI/tools/eval/trec_eval.9.0.4/trec_eval -c -M 10 -m recip_rank \
$PATH_ANSERINI/src/main/resources/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt \
splade_result.trec

$PATH_ANSERINI/tools/eval/trec_eval.9.0.4/trec_eval -c -mrecall -mmap \
$PATH_ANSERINI/src/main/resources/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt \
splade_result.trec
```
