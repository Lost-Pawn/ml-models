"""
The actual neural network. This is a standard encoder decoder setup
with Bahdanau style attention, the same architecture used in the
original neural machine translation papers before transformers took
over. GRU is used instead of LSTM to keep it a bit lighter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import SOS_TOKEN


class Encoder(nn.Module):
    """
    Reads the source sentence one word at a time and produces a
    hidden state for every position, plus a final hidden state that
    seeds the decoder.
    """

    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=2)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Attention(nn.Module):
    """
    Scores how relevant every encoder position is to the decoder's
    current step, then turns those scores into weights that sum to
    one so the decoder can focus on the right source words.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        decoder_hidden_expanded = decoder_hidden.permute(1, 0, 2).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((decoder_hidden_expanded, encoder_outputs), dim=2)))
        scores = self.v(energy).squeeze(2)
        weights = F.softmax(scores, dim=1)
        return weights


class Decoder(nn.Module):
    """
    Produces the target sentence one word at a time. At every step it
    pulls a context vector from the encoder outputs using attention,
    then combines that with its own hidden state to predict the next
    word.
    """

    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=2)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs):
        embedded = self.embedding(input_token)

        weights = self.attention(hidden, encoder_outputs)
        weights = weights.unsqueeze(1)
        context = torch.bmm(weights, encoder_outputs)

        gru_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(gru_input, hidden)

        output = torch.cat((output, context), dim=2).squeeze(1)
        prediction = self.out(output)
        return prediction, hidden


class Seq2Seq(nn.Module):
    """
    Wraps the encoder and decoder together and handles teacher
    forcing during training, feeding the decoder its own previous
    guess when it is not being teacher forced.
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        decoder_input = torch.full((batch_size, 1), SOS_TOKEN, dtype=torch.long, device=self.device)

        for t in range(trg_len):
            prediction, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t, :] = prediction

            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            top1 = prediction.argmax(1).unsqueeze(1)
            decoder_input = trg[:, t].unsqueeze(1) if use_teacher_forcing else top1

        return outputs
