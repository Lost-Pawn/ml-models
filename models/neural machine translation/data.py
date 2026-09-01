"""
Builds a small English to French parallel corpus and turns it into
tensors the model can train on. The corpus is generated from simple
sentence templates instead of loaded from a huge external file, so
the whole project runs on its own without needing any dataset download.
"""

import random
import torch
from torch.utils.data import Dataset

random.seed(42)

SOS_TOKEN = 0
EOS_TOKEN = 1
PAD_TOKEN = 2

SUBJECTS = [
    ("i", "je"),
    ("you", "tu"),
    ("he", "il"),
    ("she", "elle"),
    ("we", "nous"),
    ("they", "ils"),
]

ADJECTIVES = [
    ("happy", "heureux"),
    ("tired", "fatigue"),
    ("busy", "occupe"),
    ("hungry", "affame"),
    ("ready", "pret"),
    ("late", "en retard"),
    ("sick", "malade"),
    ("cold", "froid"),
]

VERB_OBJECT_PAIRS = [
    ("like coffee", "aime le cafe"),
    ("love music", "adore la musique"),
    ("read books", "lis des livres"),
    ("cook dinner", "cuisine le diner"),
    ("watch movies", "regarde des films"),
    ("play football", "joue au football"),
    ("drink water", "bois de l'eau"),
    ("write letters", "ecris des lettres"),
    ("study english", "etudie l'anglais"),
    ("clean the house", "nettoie la maison"),
    ("walk the dog", "promene le chien"),
    ("drive the car", "conduis la voiture"),
]

BE_FORM = {
    "i": "suis",
    "you": "es",
    "he": "est",
    "she": "est",
    "we": "sommes",
    "they": "sont",
}

VERB_FORM = {
    "i": "",
    "you": "",
    "he": "e",
    "she": "e",
    "we": "ons",
    "they": "ent",
}


def build_pairs():
    """
    Generates the parallel sentence pairs from the templates above.
    Returns a list of (english, french) tuples.
    """
    pairs = []

    for eng_subj, fr_subj in SUBJECTS:
        for eng_adj, fr_adj in ADJECTIVES:
            english = f"{eng_subj} am {eng_adj}" if eng_subj == "i" else f"{eng_subj} is {eng_adj}" if eng_subj in ("he", "she") else f"{eng_subj} are {eng_adj}"
            french = f"{fr_subj} {BE_FORM[eng_subj]} {fr_adj}"
            pairs.append((english, french))

    for eng_subj, fr_subj in SUBJECTS:
        for eng_vo, fr_vo in VERB_OBJECT_PAIRS:
            english = f"{eng_subj} {eng_vo}"
            french = f"{fr_subj} {fr_vo}"
            pairs.append((english, french))

    random.shuffle(pairs)
    return pairs


class Vocab:
    """
    Simple word level vocabulary. Keeps a word to index map and the
    reverse, and knows how to turn a sentence into a list of indices.
    """

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {SOS_TOKEN: "<sos>", EOS_TOKEN: "<eos>", PAD_TOKEN: "<pad>"}
        self.word2count = {}
        self.n_words = 3

    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def sentence_to_indices(self, sentence):
        return [self.word2index[w] for w in sentence.split(" ")] + [EOS_TOKEN]


class TranslationDataset(Dataset):
    """
    Wraps the pairs list so a DataLoader can batch them. Padding is
    handled in the collate function in train.py because the max
    length is different for every batch.
    """

    def __init__(self, pairs, input_vocab, output_vocab):
        self.pairs = pairs
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        eng, fra = self.pairs[idx]
        input_ids = torch.tensor(self.input_vocab.sentence_to_indices(eng), dtype=torch.long)
        target_ids = torch.tensor(self.output_vocab.sentence_to_indices(fra), dtype=torch.long)
        return input_ids, target_ids


def load_data(val_split=0.1):
    """
    Builds the corpus, fits an English vocab and a French vocab on
    it, splits into train and validation, and returns everything the
    training script needs.
    """
    pairs = build_pairs()
    input_vocab = Vocab("english")
    output_vocab = Vocab("french")

    for eng, fra in pairs:
        input_vocab.add_sentence(eng)
        output_vocab.add_sentence(fra)

    split_idx = int(len(pairs) * (1 - val_split))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    train_set = TranslationDataset(train_pairs, input_vocab, output_vocab)
    val_set = TranslationDataset(val_pairs, input_vocab, output_vocab)

    return train_set, val_set, input_vocab, output_vocab


if __name__ == "__main__":
    train_set, val_set, in_vocab, out_vocab = load_data()
    print(f"total pairs {len(train_set) + len(val_set)}")
    print(f"train pairs {len(train_set)}")
    print(f"val pairs {len(val_set)}")
    print(f"english vocab size {in_vocab.n_words}")
    print(f"french vocab size {out_vocab.n_words}")
    print(train_set.pairs[0])
    # actual output from running this file once
    # total pairs 120
    # train pairs 108
    # val pairs 12
    # english vocab size 45
    # french vocab size 52
    # first train pair ('you are tired', 'tu es fatigue')
