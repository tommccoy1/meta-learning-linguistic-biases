# Based on code from https://towardsdatascience.com/paper-repro-deep-metalearning-using-maml-and-reptile-fd1df1cc81b0

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import random
from random import shuffle
from collections import OrderedDict


# Redefine a basic PyTorch model to allow
# for double gradients and manual modification
# of weights
class ModifiableModule(nn.Module):
    def params(self):
        return [p for _, p in self.named_params()]

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self):
        subparams = []
        for name, mod in self.named_submodules():
            for subname, param in mod.named_params():
                subparams.append((name + '.' + subname, param))
        return self.named_leaves() + subparams

    def set_param(self, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in self.named_submodules():
                if module_name == name:
                    mod.set_param(rest, param)
                    break
        else:
            setattr(self, name, param)

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = V(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

            
    def load_state_dict(self, sdict, same_var=False):
        for name in sdict:
            param = sdict[name]
            if not same_var:
                param = V(param.data.clone(), requires_grad=True)
                
            self.set_param(name, param)
            
    def state_dict(self):
        return OrderedDict(self.named_params())

# Redefined linear layer
class GradLinear(ModifiableModule):
    def __init__(self, *args, **kwargs):
        super(GradLinear, self).__init__()
        ignore = nn.Linear(*args, **kwargs)
        self.weights = V(ignore.weight.data, requires_grad=True)
        self.bias = V(ignore.bias.data, requires_grad=True)

    def forward(self, x):
        return F.linear(x, self.weights, self.bias)

    def named_leaves(self):
        return [('weights', self.weights), ('bias', self.bias)]


# Redefined LSTM
class GradLSTM(ModifiableModule):
    def __init__(self, input_size, hidden_size):
        super(GradLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        ignore_wi = nn.Linear(hidden_size + input_size, hidden_size)
        ignore_wf = nn.Linear(hidden_size + input_size, hidden_size)
        ignore_wg = nn.Linear(hidden_size + input_size, hidden_size)
        ignore_wo = nn.Linear(hidden_size + input_size, hidden_size)
        
        self.wi_weights = V(ignore_wi.weight.data, requires_grad=True)
        self.wi_bias = V(ignore_wi.bias.data, requires_grad=True)
        self.wf_weights = V(ignore_wf.weight.data, requires_grad=True)
        self.wf_bias = V(ignore_wf.bias.data, requires_grad=True)
        self.wg_weights = V(ignore_wg.weight.data, requires_grad=True)
        self.wg_bias = V(ignore_wg.bias.data, requires_grad=True)
        self.wo_weights = V(ignore_wo.weight.data, requires_grad=True)
        self.wo_bias = V(ignore_wo.bias.data, requires_grad=True)
        
        
    def forward(self, inp, hidden):
        hx, cx = hidden
        input_plus_hidden = torch.cat((inp, hx), 2)
        
        i_t = torch.sigmoid(F.linear(input_plus_hidden, self.wi_weights, self.wi_bias))
        f_t = torch.sigmoid(F.linear(input_plus_hidden, self.wf_weights, self.wf_bias))
        g_t = torch.tanh(F.linear(input_plus_hidden, self.wg_weights, self.wg_bias))
        o_t = torch.sigmoid(F.linear(input_plus_hidden, self.wo_weights, self.wo_bias))
        
        cx = f_t * cx + i_t * g_t
        hx = o_t * torch.tanh(cx)
        
        return hx, (hx, cx)

    
    def named_leaves(self):
        return [('wi_weights', self.wi_weights), ('wi_bias', self.wi_bias), 
                ('wf_weights', self.wf_weights), ('wf_bias', self.wf_bias),
                ('wg_weights', self.wg_weights), ('wg_bias', self.wg_bias),
                ('wo_weights', self.wo_weights), ('wo_bias', self.wo_bias)]



# Redefined embedding layer
class GradEmbedding(ModifiableModule):
    def __init__(self, *args, **kwargs):
        super(GradEmbedding, self).__init__()
        ignore = nn.Embedding(*args, **kwargs)
        self.weights = V(ignore.weight.data, requires_grad=True)
        
        
    def forward(self, x):
        return F.embedding(x, self.weights)
    
    def named_leaves(self):
        return [('weights', self.weights)]

# Encoder/decoder model
class EncoderDecoder(ModifiableModule):
    def __init__(self, vocab_size, input_size, hidden_size):
        super(EncoderDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.embedding = GradEmbedding(vocab_size, input_size)
        self.enc_lstm = GradLSTM(input_size, hidden_size)
        
        self.dec_lstm = GradLSTM(input_size, hidden_size)
        self.dec_output = GradLinear(hidden_size, vocab_size)
        
        self.max_length = 20
        
        
    def forward(self, sequence_list):
        # Initialize the hidden and cell states
        hidden = (V(torch.zeros(1, len(sequence_list), self.hidden_size)), 
                  V(torch.zeros(1, len(sequence_list), self.hidden_size))) 
        
        # The input is a list of sequences. Here the sequences are converted
        # into integer keys
        all_seqs = []
        for sequence in sequence_list:
            this_seq = []
            # Iterate over the sequence
            for elt in sequence:
                ind = self.char2ind[elt]
                this_seq.append(ind)
            all_seqs.append(torch.LongTensor(this_seq))

        max_length = max([len(x) for x in sequence_list])
        if max_length > 0:    
            # Pad the sequences to allow batching 
            all_seqs = torch.nn.utils.rnn.pad_sequence(all_seqs)

            all_seqs_onehot = (all_seqs > 0).type(torch.FloatTensor)
        
            # Pass the sequences through the encoder, one character at a time
            for index, elt in enumerate(all_seqs):
                # Embed the character
                emb = self.embedding(elt.unsqueeze(0))

                # Pass through the LSTM
                output, hidden_new = self.enc_lstm(emb, hidden)
                hidden_prev = hidden

                # Awkward solution to variable length inputs: For each sequence in the batch, use the
                # new hidden state if the sequence is still being updated, or retain the old
                # hidden state if the sequence is over and we're now in the padding
                hx = hidden_prev[0] * (1 - all_seqs_onehot[index].unsqueeze(0).unsqueeze(2).expand(hidden_prev[0].shape)) + hidden_new[0] * all_seqs_onehot[index].unsqueeze(0).unsqueeze(2).expand(hidden_prev[0].shape)
                cx = hidden_prev[1] * (1 - all_seqs_onehot[index].unsqueeze(0).unsqueeze(2).expand(hidden_prev[1].shape)) + hidden_new[1] * all_seqs_onehot[index].unsqueeze(0).unsqueeze(2).expand(hidden_prev[1].shape)
            
                hidden = (hx, cx)
        
        # Decoding

        # Previous output characters (used as input for the following time step)
        prev_output = ["SOS" for _ in range(len(sequence_list))]

        # Accumulates the output sequences
        out_strings = ["" for _ in range(len(sequence_list))]

        # Probabilities at each output position (used for computing the loss)
        logits = []
        
        for i in range(self.max_length):
            # Determine the previous output character for each element
            # of the batch; to be used as the input for this time step
            prev_outputs = []
            for elt in prev_output:
                ind = self.char2ind[elt]
                prev_outputs.append(ind)
                
            # Embed the previous outputs
            emb = self.embedding(torch.LongTensor([prev_outputs]))

            # Pass through the decoder
            output, hidden = self.dec_lstm(emb, hidden)

            # Determine the output probabilities used to make predictions
            pred = self.dec_output(output)
            probs = F.log_softmax(pred, dim=2)
            logits.append(probs)

            # Discretize the output labels (via argmax) for generating an output character
            topv, topi = probs.data.topk(1)
            label = topi[0] 
            
            prev_output = []
            for index, elt in enumerate(label):
                char = self.ind2char[elt.item()]
                
                out_strings[index] += char
                prev_output.append(char)
        
        return out_strings, logits
        
    def named_submodules(self):
        return [('embedding', self.embedding), ('enc_lstm', self.enc_lstm),
                ('dec_lstm', self.dec_lstm), ('dec_output', self.dec_output)]
        
    # Create a copy of the model
    def create_copy(self, same_var=False):
        new_model = EncoderDecoder(self.vocab_size, self.input_size, self.hidden_size)
        new_model.copy(self, same_var=same_var)

        return new_model
        
    def set_dicts(self, vocab_list):
        vocab_list = ["NULL", "SOS", "EOS"] + vocab_list

        index = 0
        char2ind = {}
        ind2char = {}

        for elt in vocab_list:
            char2ind[elt] = index
            ind2char[index] = elt
            index += 1

        self.char2ind = char2ind
        self.ind2char = ind2char

