import torch
import torch.nn as nn


# https://arxiv.org/pdf/1511.08630.pdf
class CLSTM(nn.Module):
    def __init__(self, wordvectors):
        super(CLSTM, self).__init__()

        hidden_dim = 150
        lstm_layers = 1
        kernel_size = 3
        num_filters = 150  # equals out_channels in Conv
        dropout_rate = 0.5

        # word embedding
        self.embedding = nn.Embedding.from_pretrained(wordvectors)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # CNN layer
        self.conv = nn.Conv2d(1, num_filters, kernel_size=(kernel_size, 300))

        # LSTM
        self.lstm = nn.LSTM(num_filters, hidden_dim, num_layers=lstm_layers,
                            dropout=dropout_rate)

        # softmax layer after lstm layer
        self.softmax = nn.Softmax()

        # Label Output
        self.hidden2label = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        embed = self.embedding(x)

        c_out = embed
        c_out = torch.transpose(c_out, 0, 1)
        c_out = self.dropout(c_out)
        c_out = self.conv(c_out)
        c_out = nn.functional.relu(c_out)
        c_out = torch.transpose(c_out, 0, 1)  # we get a tensor with the shape [54, 4, 1]
        c_out = torch.transpose(c_out, 1, 2)
        # print(c_out.shape)

        l_out, hidden = self.lstm(c_out)
        output = hidden[-1]

        # disabled softmax, because accuracy gets higher without (from 23% to 39%)
        # output = self.softmax(output)

        output = self.hidden2label(output)
        output = output.squeeze(0)
        return output
