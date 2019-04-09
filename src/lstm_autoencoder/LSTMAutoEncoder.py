import torch
from torch import nn, optim
import torch.nn.functional as F

class LSTMAutoEncoder(nn.Module):
    def __init__(self, out_len=3):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = nn.LSTM(input_size=1, hidden_size=50, num_layers=3)
        self.decoder = nn.LSTMCell(input_size=50, hidden_size=50)
        self.linear = nn.Linear(50, 1)
        self.out_len = out_len
        
    def forward(self, inputs):
        """
        :param inputs:(batch_size, seq_len, vec_len)
        :return:
        """
        batch_size = inputs.shape[0]
        outputs = torch.zeros((batch_size, self.out_len), device=inputs.device)
        hide_out, (h, c) = self.encoder(inputs.permute(1, 0, 2))
        h = h[-1, ...]
        c = c[-1, ...]
        for i in range(self.out_len):
            cur_input = self.attention(hide_out, h)
            h, c = self.decoder(cur_input, hx=(h, c))
            outputs[:, i] = self.linear(h).view(-1)
        return outputs
    
    @staticmethod
    def attention(encoder_hide, cur_hide):
        dist = torch.sum(encoder_hide * cur_hide[None], dim=-1)
        wt = F.softmax(dist, dim=0)
        cur_input = torch.sum(wt[..., None] * encoder_hide, dim=0)
        return cur_input