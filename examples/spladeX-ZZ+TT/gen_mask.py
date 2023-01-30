from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import json
from string import ascii_letters, digits, punctuation


def check_eng(s):
  for i in s:
    if i not in ascii_letters+digits+punctuation:
      return False
  return True
def addToMask(tks, mask, vocab_dict):
  for id in tks:
    if check_eng(vocab_dict[id]):
      mask[id] = vocab_dict[id]
def main():
    dataset = load_dataset("crystina-z/mmarco-corpus")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
    vocab_dict = tokenizer.get_vocab()
    vocab_dict = {v: k for k, v in vocab_dict.items()}
    loader = DataLoader(dataset['train'], batch_size=16) 
    mask = {}
    tokenized_dataset = dataset['train'].map(lambda x: tokenizer(x['text'], truncation=True), batched =True)
    td = tokenized_dataset.map(lambda x: addToMask(x['input_ids'],mask, vocab_dict))
    with open("mask.txt", "w") as document1:
        document1.write(json.dumps(mask))


if __name__ == "__main__":
    main()
