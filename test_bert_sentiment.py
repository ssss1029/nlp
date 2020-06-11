"""
Fine-grained sentiment analysis on SST-5 using Bert
From: https://github.com/munikarmanish/bert-sentiment
"""

import argparse
import os
import torch

from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from tqdm import tqdm

from datasets.SSTDataset import SSTDataset

def rpad(array, n=70):
    """Right padding."""
    current_len = len(array)
    if current_len > n:
        return array[: n - 1]
    extra = n - current_len
    return array + ([0] * extra)


def main(weights):

    config = BertConfig.from_pretrained("bert-large-uncased")
    config.num_labels = 5
    model = BertForSequenceClassification.from_pretrained("bert-large-uncased", config=config).cuda()
    model = torch.nn.DataParallel(model, dim=0)
    model.load_state_dict(torch.load(weights))

    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

    print("Beginning testing")

    input_sentence = input("Input sentence: ")
    while input_sentence != "q":
        batch = torch.tensor(rpad(tokenizer.encode('[CLS] ' + input_sentence + ' [SEP]'))).unsqueeze(0)
        print(batch)
        logits = model(batch)[0]
        sentiment_softmax = torch.nn.functional.softmax(logits)
        print(sentiment_softmax)
        
        # Next iteration
        input_sentence = input("Input sentence: ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bert on SST-5')
    parser.add_argument('--weights', default='checkpoints/TEMP/bert-large-uncased__sst5__fine.pth', type=str)

    args = parser.parse_args()
    argsdict = vars(args)

    print(argsdict)
    main(**argsdict)
