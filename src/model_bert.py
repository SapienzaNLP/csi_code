import abc
from math import sqrt
from typing import Union, List

import torch
from torch.nn import LSTM, Dropout, Linear
from torch.nn import Module, Parameter
from torch.nn.functional import softmax

if torch.cuda.is_available():
    FloatTensorType = torch.cuda.FloatTensor
    LongTensorType = torch.cuda.LongTensor
else:
    FloatTensorType = torch.FloatTensor
    LongTensorType = torch.LongTensor


class BaseContextualEmbedder(torch.nn.Module, metaclass=abc.ABCMeta):
    '''This class is taken from QBERT, by Michele Bevilacqua and Roberto Navigli.
     Discover more at https://github.com/mbevila/qbert'''

    embedding_dim: int
    n_hidden_states: int
    retrain_model: bool

    def __init__(self, retrain_model: bool = False):
        super().__init__()
        self.retrain_model = retrain_model

    def forward(
            self,
            src_tokens: Union[None, torch.LongTensor] = None,
            src_tokens_str: Union[None, List[List[str]]] = None,
            batch_major: bool = True,
            **kwargs
    ):
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    @property
    def embed_dim(self):
        return self.embedding_dim

    @property
    def embedded_dim(self):
        return self.embedding_dim

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return super().state_dict(destination, prefix, keep_vars)

class BERTEmbedder(BaseContextualEmbedder):
    '''This class is taken from QBERT, by Michele Bevilacqua and Roberto Navigli.
         Discover more at https://github.com/mbevila/qbert'''

    DEFAULT_MODEL = 'bert-base-cased'

    @staticmethod
    def _do_imports():
        import pytorch_pretrained_bert as bert
        return bert

    def __init__(
        self,
        name: Union[str, None] = None,
        weights: str = "",
        retrain_model=False,
        layers=(-1,)
    ):
        assert not retrain_model
        super(BERTEmbedder, self).__init__(retrain_model=False)

        assert not retrain_model

        if not name:
            name = self.DEFAULT_MODEL

        self.retokenize = True
        self.realign = 'MEAN'
        self.uncased = 'uncased' in name
        self.layers = layers

        bert = self._do_imports()
        self.bert_tokenizer = bert.BertTokenizer.from_pretrained(name)
        self.bert_model = bert.BertModel.from_pretrained(name)

        if weights:
            state = torch.load(weights)['state']
            state = {".".join(k.split('.')[1:]): v for k, v in state.items() if k.startswith('bert.')}
            self.bert_model.load_state_dict(state)

        for par in self.parameters():
            par.requires_grad = False

    def _subtokenize_sequence(self, tokens):
        split_tokens = []
        merge_to_previous = []
        for token in tokens:
            for i, sub_token in enumerate(self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)):
                split_tokens.append(sub_token)
                merge_to_previous.append(i > 0)

        split_tokens = ['[CLS]'] + split_tokens + ['[SEP]']
        return split_tokens, merge_to_previous

    def _convert_to_indices(self, subtokens, maxlen=-1):
        unpadded_left = [self.bert_tokenizer.vocab[st] for st in subtokens[:-1]]
        if maxlen > 0:
            unpadded_left = unpadded_left[:maxlen]
            unpadded_left = unpadded_left +  [self.bert_tokenizer.vocab['[SEP]']] + [0] * (maxlen - len(unpadded_left) - 1)
        return torch.LongTensor(unpadded_left)

    def forward(self, src_tokens_str: List[List[str]], src_tokens=None, batch_major=True, padding_str='<pad>',
                **kwargs):
        output_all_encoded_layers = False
        if "output_all_encoded_layers" in kwargs:
            output_all_encoded_layers = kwargs["output_all_encoded_layers"]
        if "layers_to_sum" in kwargs:
            layers_to_sum = kwargs["layers_to_sum"]
        else:
            layers_to_sum = [-1]
        if "concat" in kwargs:
            concatenate = kwargs["concat"]
        else:
            concatenate = False
        if "layers_to_cat" in kwargs:
            layers_to_cat = kwargs["layers_to_cat"]
        else:
            layers_to_cat = [-1]

        if self.retokenize:
            src_tokens_str = [[t for t in seq if t != '<pad>'] for seq in src_tokens_str]
            subtoken_str = []
            merge_to_previous = []
            for seq in src_tokens_str:
                ss, mm = self._subtokenize_sequence(seq)
                subtoken_str.append(ss)
                merge_to_previous.append(mm)

            max_token_len = max(map(len, src_tokens_str))
            subtoken_len = list(map(len, subtoken_str))
            max_subtoken_len = max(subtoken_len)

            if concatenate:
                contextual_embeddings = torch.zeros(len(src_tokens_str), max_token_len,
                                                    self.embedding_dim * len(layers_to_cat),
                                                    dtype=torch.float32).to(self.device)
            else:
                contextual_embeddings = torch.zeros(len(src_tokens_str), max_token_len, self.embedding_dim,
                                                    dtype=torch.float32).to(self.device)
            input_ids = torch.stack([self._convert_to_indices(seq, max_subtoken_len) for seq in subtoken_str],
                                    dim=0).to(self.device)
            attention_mask = input_ids > 0
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.int64)
            # for i, l in enumerate(subtoken_len):
            #     token_type_ids[i, l - 1:] = 1
            raw_output, _ = self.bert_model.forward(input_ids=input_ids, token_type_ids=token_type_ids,
                                                    attention_mask=attention_mask,
                                                    output_all_encoded_layers=output_all_encoded_layers)
            if output_all_encoded_layers:
                if not concatenate:
                    raw_output = torch.stack([raw_output[x] for x in layers_to_sum])
                    raw_output = torch.sum(raw_output.permute(1, 0, 2, 3), 1)  # sentence  x layers x tokens x dim
                else:
                    raw_output = torch.cat([raw_output[x] for x in layers_to_cat], -1)

            raw_output = raw_output[:, 1:]

            if self.realign == 'MEAN':
                for n_seq, merge_seq in enumerate(merge_to_previous):
                    n_token = -1
                    n_subtoken_per_token = 1
                    for n_subtoken, merge in enumerate(merge_seq):
                        if merge:
                            n_subtoken_per_token += 1
                        else:
                            n_subtoken_per_token = 1
                            n_token += 1

                        prev = contextual_embeddings[n_seq, n_token]
                        next = raw_output[n_seq, n_subtoken]
                        contextual_embeddings[n_seq, n_token] = prev + (next - prev) / n_subtoken_per_token

            elif self.realign == 'FIRST':
                for n_seq, merge_seq in enumerate(merge_to_previous):

                    ids = [i for i, merge in enumerate(merge_seq) if not merge]

                    for n_token, n_subtoken in enumerate(ids):
                        contextual_embeddings[n_seq, n_token] = raw_output[n_seq, n_subtoken]

            else:
                raise

        if not batch_major:
            contextual_embeddings = contextual_embeddings.transpose(0, 1)

        return {'inner_states': [contextual_embeddings]}

    @property
    def embedding_dim(self):
        return self.bert_model.encoder.layer[-1].output.dense.out_features

    @property
    def n_hidden_states(self):
        return 1

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

class Attention(Module):

    def __init__(self, in_features):
        super(Attention, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(FloatTensorType(1, in_features))
        stdv = 1. / sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        h = torch.transpose(inputs, 1, 2)
        u = self.tanh(h)
        u = torch.matmul(self.weight, u)
        a = softmax(u, dim=2)
        a = torch.transpose(a, 1, 2)
        c = torch.matmul(h, a)
        c = c.transpose(1, 2)
        ones = torch.ones((inputs.size(0), inputs.size(1),1)).type(FloatTensorType)
        c = torch.matmul(ones, c)

        return torch.cat((inputs, c), dim=2)

class BertDense(Module):
    def __init__(self, output_vocabulary_size, bert_name='bert-large-cased'):
        super(BertDense, self).__init__()
        self.bert = BERTEmbedder(bert_name)
        self.output = Linear(in_features=self.bert.embedding_dim, out_features=output_vocabulary_size)
        if torch.cuda.is_available():
           self.cuda()

    def forward(self, x):
        with torch.no_grad():
            x = self.bert.eval()(x)['inner_states'][-1]
        x = self.output(x)
        x = torch.log_softmax(x, dim=-1)
        return x

class BertLSTM(Module):

    def __init__(self, output_vocabulary_size, bert_name='bert-base-uncased',
                 lstm_units_size=512, lstm_layers = 2, dropout_rate = 0.1):

        super(BertLSTM, self).__init__()
        self.bert = BERTEmbedder(bert_name)
        self.lstm = LSTM(input_size=self.bert.embedding_dim, hidden_size=lstm_units_size,
                         num_layers=lstm_layers, bidirectional=True, batch_first=True)
        self.dropout = Dropout(p=dropout_rate)
        self.attention = Attention(in_features= lstm_units_size * 2)
        next_layer_in_features = lstm_units_size * 4
        self.output = Linear(in_features=next_layer_in_features, out_features=output_vocabulary_size)
        if torch.cuda.is_available():
           self.cuda()

    def forward(self, x):
        lengths = [len(seq) for seq in x]
        with torch.no_grad():
            x = self.bert.eval()(x)['inner_states'][-1]
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False
        )
        x, _ = self.lstm(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        if not self.dropout is None:
            x = self.dropout(x)
        if not self.attention is None:
            x = self.attention(x)
        x = self.output(x)
        x = torch.log_softmax(x, dim=-1)
        return x
