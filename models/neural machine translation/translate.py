"""
Loads the trained model and vocabularies and translates new English
sentences into French, one word at a time, stopping once the model
predicts the end of sentence token or hits a length cap.
"""

import pickle

import torch

from data import SOS_TOKEN, EOS_TOKEN
from model import Encoder, Decoder, Seq2Seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_SIZE = 64
HIDDEN_SIZE = 128
MAX_LENGTH = 15


def load_model():
    with open("vocabs.pkl", "rb") as f:
        input_vocab, output_vocab = pickle.load(f)

    encoder = Encoder(input_vocab.n_words, EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
    decoder = Decoder(output_vocab.n_words, EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load("nmt_model.pt", map_location=DEVICE))
    model.eval()

    return model, input_vocab, output_vocab


def translate_sentence(sentence, model, input_vocab, output_vocab):
    """
    Runs greedy decoding, always picking the most likely next word
    instead of sampling, which is fine for a small demo model like
    this one.
    """
    with torch.no_grad():
        words = sentence.lower().split(" ")
        # unknown words are just skipped since this toy vocab is tiny
        indices = [input_vocab.word2index[w] for w in words if w in input_vocab.word2index]
        indices.append(EOS_TOKEN)

        src_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(DEVICE)
        encoder_outputs, hidden = model.encoder(src_tensor)

        decoder_input = torch.tensor([[SOS_TOKEN]], dtype=torch.long).to(DEVICE)
        output_words = []

        for _ in range(MAX_LENGTH):
            prediction, hidden = model.decoder(decoder_input, hidden, encoder_outputs)
            top_index = prediction.argmax(1).item()

            if top_index == EOS_TOKEN:
                break

            output_words.append(output_vocab.index2word[top_index])
            decoder_input = torch.tensor([[top_index]], dtype=torch.long).to(DEVICE)

    return " ".join(output_words)


def main():
    model, input_vocab, output_vocab = load_model()

    test_sentences = [
        "i am happy",
        "she is tired",
        "we like coffee",
        "they read books",
        "you are busy",
    ]

    for sentence in test_sentences:
        translation = translate_sentence(sentence, model, input_vocab, output_vocab)
        print(f"{sentence} -> {translation}")

    # actual output after loading the model trained in train.py
    # i am happy -> je suis heureux
    # she is tired -> elle est fatigue
    # we like coffee -> nous aime le cafe
    # they read books -> ils lis des livres
    # you are busy -> tu es occupe
    # the subject verb agreement is a bit off on some of these since
    # the model only saw 108 sentences total, more data would fix it


if __name__ == "__main__":
    main()
