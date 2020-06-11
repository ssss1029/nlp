"""
Fine-grained sentiment analysis on SST-5 using Bert
From: https://github.com/munikarmanish/bert-sentiment
"""

import argparse
import os
import torch

from transformers import BertConfig, BertForSequenceClassification
from tqdm import tqdm

from datasets.SSTDataset import SSTDataset

def train_one_epoch(model, lossfn, optimizer, dataset, batch_size=32):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for batch, labels in tqdm(generator):
        batch, labels = batch.cuda(), labels.cuda()

        optimizer.zero_grad()
        logits = model(batch)[0]
        # import pdb
        # pdb.set_trace() 
        # print(logits)
        loss = lossfn(logits, labels)

        # print(loss)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred_labels = torch.argmax(logits, axis=1)
        train_acc += (pred_labels == labels).sum().item()

        # print(batch, labels)
        # exit()
    
    train_loss /= len(dataset)
    train_acc /= len(dataset)
    
    return train_loss, train_acc

def evaluate(model, lossfn, optimizer, dataset, batch_size=32):
    generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.eval()
    loss, acc = 0.0, 0.0
    with torch.no_grad():
        for batch, labels in tqdm(generator):
            batch, labels = batch.cuda(), labels.cuda()
            logits = model(batch)[0]
            error = lossfn(logits, labels)
            loss += error.item()
            pred_labels = torch.argmax(logits, axis=1)
            acc += (pred_labels == labels).sum().item()
    loss /= len(dataset)
    acc /= len(dataset)
    return loss, acc


def main(num_epochs, batch_size, save_dir):

    if os.path.exists(save_dir):
        resp = None
        while resp not in {"yes", "no", "y", "n"}:
            resp = input(f"{save_dir} already exists. Overwrite contents? [y/n]: ")
            if resp == "yes" or resp == "y":
                break
            elif resp == "no" or resp =="n":
                print("Exiting")
                exit()
    else:
        os.makedirs(save_dir, exist_ok=True)
    
    train_data = SSTDataset("train", root=True, binary=False)
    dev_data = SSTDataset("train", root=True, binary=False)
    test_data = SSTDataset("train", root=True, binary=False)

    config = BertConfig.from_pretrained("bert-large-uncased")
    config.num_labels = 5
    model = BertForSequenceClassification.from_pretrained("bert-large-uncased", config=config).cuda()
    model = torch.nn.DataParallel(model, dim=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    lossfn = torch.nn.CrossEntropyLoss()

    print("Beginning tuning")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, lossfn, optimizer, train_data, batch_size=batch_size
        )
        # val_loss, val_acc = evaluate(
        #     model, lossfn, optimizer, devset, batch_size=batch_size
        # )
        test_loss, test_acc = evaluate(
            model, lossfn, optimizer, test_data, batch_size=batch_size
        )

        print(f"Train Loss: {train_loss} | Test Loss: {test_loss} | Test Acc: {test_acc}")

        if save_dir != None:
            print("Saving Model")
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "bert-large-uncased__sst5__fine.pth")
            )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bert on SST-5')
    parser.add_argument('--num-epochs', default=1, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--save-dir', default='checkpoints/TEMP/', type=str)

    args = parser.parse_args()
    argsdict = vars(args)

    print(argsdict)
    main(**argsdict)
