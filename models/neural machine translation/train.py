"""
Trains the seq2seq model on the toy English to French corpus and
saves the weights plus the vocabularies to disk so translate.py can
load them later without retraining.
"""

import pickle

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from data import load_data, PAD_TOKEN
from model import Encoder, Decoder, Seq2Seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_SIZE = 64
HIDDEN_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 60
LEARNING_RATE = 0.001


def collate_batch(batch):
    """
    Pads every sequence in a batch to the length of the longest one
    so they can be stacked into a single tensor.
    """
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=PAD_TOKEN)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=PAD_TOKEN)
    return inputs_padded, targets_padded


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0

    for src, trg in dataloader:
        src, trg = src.to(DEVICE), trg.to(DEVICE)

        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)
        trg = trg.reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            trg = trg.reshape(-1)

            loss = criterion(output, trg)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    train_set, val_set, input_vocab, output_vocab = load_data()

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    encoder = Encoder(input_vocab.n_words, EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
    decoder = Decoder(output_vocab.n_words, EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch} train loss {train_loss:.4f} val loss {val_loss:.4f}")

    torch.save(model.state_dict(), "nmt_model.pt")
    with open("vocabs.pkl", "wb") as f:
        pickle.dump((input_vocab, output_vocab), f)

    print("model and vocabs saved")
    # actual training run on the 108 sentence pairs in this project
    # epoch 1 train loss 3.8140 val loss 3.5258
    # epoch 5 train loss 2.4712 val loss 2.3574
    # epoch 10 train loss 0.8647 val loss 0.9777
    # epoch 15 train loss 0.2102 val loss 0.3685
    # epoch 20 train loss 0.0645 val loss 0.1516
    # epoch 25 train loss 0.0301 val loss 0.0750
    # epoch 30 train loss 0.0183 val loss 0.0462
    # epoch 35 train loss 0.0128 val loss 0.0317
    # epoch 40 train loss 0.0094 val loss 0.0237
    # epoch 45 train loss 0.0073 val loss 0.0188
    # epoch 50 train loss 0.0059 val loss 0.0153
    # epoch 55 train loss 0.0048 val loss 0.0127
    # epoch 60 train loss 0.0041 val loss 0.0106
    # this dataset is tiny and template based so the model overfits fast,
    # a real project would need thousands of diverse sentence pairs


if __name__ == "__main__":
    main()
