""" RNN class wrapper, taking symbols, producing vector representations of prefixes

Sort of a vestigial part of more complex older code, but here in case we'd like
to hand-write RNNs again.
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import math

from transformers import BertConfig, BertForMaskedLM
from transformers.models.bert.modeling_bert import (
    BertSelfAttention,
    BertAttention,
    BertLayer,
    BertIntermediate,
    BertEncoder,
    BertModel,
    BertEmbeddings,
    BertPooler,
    BertLMPredictionHead,
    BertOnlyMLMHead,
)

MAX_LEN = 6000


class BertSelfOutputCustom(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.attn_output_fc:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.dense = None
        if config.layer_norm:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.LayerNorm = nn.Identity()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.residual = config.residual
        self.attn_output_fc = config.attn_output_fc

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.attn_output_fc:
            hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.residual:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertSelfAttentionCustom(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        if config.freeze_uniform_attention:
            self.key.weight.data = torch.zeros((config.hidden_size, config.hidden_size))
            self.key.weight.requires_grad = False
            self.key.bias.data = torch.zeros((config.hidden_size,))
            self.key.bias.requires_grad = False

            self.query.weight.data = torch.zeros((config.hidden_size, config.hidden_size))
            self.query.weight.requires_grad = False
            self.query.bias.data = torch.zeros((config.hidden_size,))
            self.query.bias.requires_grad = False

        if config.freeze_id_value_matrix:
            self.value.weight.data = torch.eye(config.hidden_size, requires_grad=False)
            self.value.weight.requires_grad = False
            self.value.bias.data = torch.zeros((config.hidden_size,))
            self.value.bias.requires_grad = False

        if config.freeze_block_value_matrix:
            block = [[0.0] * config.hidden_size] * config.hidden_size
            for i in range(4, config.hidden_size):
                for j in range(4, config.hidden_size):
                    if (i-4) // 10 == (j-4) // 10:
                        block[i][j] = 1.0
                    else:
                        block[i][j] = 0.0

            self.value.weight.data = torch.tensor(block, requires_grad=False)
            self.value.weight.requires_grad = False
            self.value.bias.data = torch.zeros((config.hidden_size,))
            self.value.bias.requires_grad = False

        if config.zero_init_attn:
            self.key.weight.data = config.zero_init_noise * torch.rand((config.hidden_size, config.hidden_size))
            self.key.weight.requires_grad = True
            self.key.bias.data = torch.zeros((config.hidden_size,))
            self.key.bias.requires_grad = True

            self.query.weight.data = config.zero_init_noise * torch.rand((config.hidden_size, config.hidden_size))
            self.query.weight.requires_grad = True
            self.query.bias.data = torch.zeros((config.hidden_size,))
            self.query.bias.requires_grad = True

            self.value.weight.data = config.zero_init_noise * torch.rand((config.hidden_size, config.hidden_size))
            self.value.weight.requires_grad = True
            self.value.bias.data = torch.zeros((config.hidden_size,))
            self.value.bias.requires_grad = True


class BertAttentionCustom(BertAttention):
    def __init__(self, config, position_embedding_type=None,):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.self = BertSelfAttentionCustom(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutputCustom(config)
        self.pruned_heads = set()


class BertOutputCustom(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        if config.layer_norm:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.LayerNorm = nn.Identity()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.residual = config.residual

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.residual:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLayerCustom(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttentionCustom(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttentionCustom(config, position_embedding_type="absolute")
        self.bert_intermediate = config.bert_intermediate
        if config.bert_intermediate:
            self.intermediate = BertIntermediate(config)
        else:
            self.intermediate = None
        self.bert_output = config.bert_output
        if config.bert_output:
            self.output = BertOutputCustom(config)
        else:
            self.output = None

    def feed_forward_chunk(self, attention_output):
        if self.bert_intermediate:
            intermediate_output = self.intermediate(attention_output)
        else:
            intermediate_output = attention_output
        if self.bert_output:
            layer_output = self.output(intermediate_output, attention_output)
        else:
            layer_output = intermediate_output
        return layer_output


class BertEncoderCustom(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList([BertLayerCustom(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False


class BertModelCustom(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoderCustom(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        # self.post_init()  # commented out because it disrupts our special initialization


class BertLMPredictionHeadCustom(BertLMPredictionHead):
    def __init__(self, config):
        super().__init__(config)
        self.bert_head_transform = config.bert_head_transform
        if not config.bert_head_transform:
            self.transform = None
        if config.freeze_decoder_to_I:
            one_hot_embeddings = torch.zeros((config.vocab_size, config.hidden_size))
            for i in range(config.vocab_size):
                one_hot_embeddings[i][i] = 1.0
            self.decoder.weight.data = one_hot_embeddings
            self.decoder.weight.requires_grad = False
            self.decoder.bias.data = torch.zeros((config.vocab_size,))
            self.decoder.bias.requires_grad = False

    def forward(self, hidden_states):
        if self.bert_head_transform:
            hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHeadCustom(BertOnlyMLMHead):
    def __init__(self, config):
        super().__init__(config)
        self.predictions = BertLMPredictionHeadCustom(config)


class BertForMaskedLMCustom(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModelCustom(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHeadCustom(config)

        # Initialize weights and apply final processing
        # self.post_init()  # commented out because it disrupts our special initialization


class PytorchTransformerModel(nn.Module):
    """
    Class for mapping sequences of symbols to sequences
    of vectors representing prefixes, using PyTorch
    RNN classes.
    """

    def __init__(self, args):
        super(PytorchTransformerModel, self).__init__()
        num_layers = args['lm']['num_layers']
        self.input_size = args['lm']['embedding_dim']
        self.hidden_size = args['lm']['hidden_dim']
        self.vocab_size = args['language']['vocab_size']
        self.n_heads = args['lm']['num_heads']
        self.e_type = args['lm']['embedding_type']
        self.token_embedding_type = args['lm']['token_embedding_type']
        self.objective = args['training']['objective']
        self.p_drop = args['training']['dropout']
        self.lm_type = args['lm']['lm_type']
        self.residual = args['lm']['residual']
        self.attn_output_fc = args['lm']['attn_output_fc']
        self.bert_intermediate = args['lm']['bert_intermediate']
        self.bert_output = args['lm']['bert_output']
        self.bert_head_transform = args['lm']['bert_head_transform']
        self.layer_norm = args['lm']['layer_norm']
        self.freeze_uniform_attention = args['lm']['freeze_uniform_attention']
        self.freeze_id_value_matrix = args['lm']['freeze_id_value_matrix']
        self.freeze_block_value_matrix = args['lm']['freeze_block_value_matrix']
        self.freeze_decoder_to_I = args['lm']['freeze_decoder_to_I']
        self.zero_init_attn = args['training']['zero_init_attn']
        self.zero_init_emb_dec = args['training']['zero_init_emb_dec']
        self.zero_init_noise = args['training']['zero_init_noise']
        self.device = args['device']

        if args['lm']['lm_type'] == 'BertForMaskedLM':
            config = BertConfig(
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_size,
                num_hidden_layers=num_layers,
                num_attention_heads=self.n_heads,
                intermediate_size=self.hidden_size,
                hidden_dropout_prob=self.p_drop,
                attention_probs_dropout_prob=self.p_drop,
                max_position_embeddings=MAX_LEN,
            )
            self.model = BertForMaskedLM(config)
        elif args['lm']['lm_type'] == 'BertForMaskedLMCustom':
            config = BertConfig(
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_size,
                num_hidden_layers=num_layers,
                num_attention_heads=self.n_heads,
                intermediate_size=self.hidden_size,
                hidden_dropout_prob=self.p_drop,
                attention_probs_dropout_prob=self.p_drop,
                max_position_embeddings=MAX_LEN,
                residual=self.residual,
                attn_output_fc=self.attn_output_fc,
                bert_intermediate=self.bert_intermediate,
                bert_output=self.bert_output,
                bert_head_transform=self.bert_head_transform,
                layer_norm=self.layer_norm,
                freeze_uniform_attention=self.freeze_uniform_attention,
                freeze_id_value_matrix=self.freeze_id_value_matrix,
                freeze_block_value_matrix=self.freeze_block_value_matrix,
                freeze_decoder_to_I=self.freeze_decoder_to_I,
                zero_init_attn=self.zero_init_attn,
                zero_init_emb_dec=self.zero_init_emb_dec,
                zero_init_noise=self.zero_init_noise,
            )
            self.model = BertForMaskedLMCustom(config)
        else:
            raise NotImplementedError('Model not supported.')

        print('config', config)
        print('model', self.model)

        self.model.to(self.device)
        tqdm.write('Constructing a {} pytorch model w hidden size {}, layers {}, dropout {}'.format(
            args['lm']['lm_type'],
            self.hidden_size,
            num_layers,
            self.p_drop,
        ))

        if args['lm']['lm_type'] in {'BertForMaskedLM', 'BertForMaskedLMCustom'}:
            position_embeddings = self.model.bert.embeddings.position_embeddings
        else:
            raise NotImplementedError('Model not supported.')

        if self.e_type == 'cos':
            funcs = [math.sin, math.cos]
            position_embeddings.weight.data = torch.tensor(
                [[funcs[i % 2](pos / 10000 ** (2 * i / self.hidden_size))
                  for i in range(self.hidden_size)]
                 for pos in range(MAX_LEN)]
            ).to(self.device)
            position_embeddings.weight.requires_grad = False

        if self.e_type == 'p' or self.e_type == 'pw':
            position_embeddings.weight.data.zero_()
            position_embeddings.weight.requires_grad = False

            self.embedding = nn.Embedding(self.vocab_size, self.input_size - 1)
            self.embedding.to(self.device)

            self.embedding_p = nn.Embedding(MAX_LEN, 1)
            self.embedding_p.weight.data = torch.tensor([[i / MAX_LEN] for i in range(MAX_LEN)])
            self.embedding_p.weight.requires_grad = False
            self.embedding_p.to(self.device)

        if self.e_type == 'pw':
            k = args['language']['bracket_types']
            self.embedding = nn.Embedding(self.vocab_size, self.input_size - 1 - k - 4)
            self.embedding_e = nn.Embedding(self.vocab_size, k + 4)

            def get_row(i):
                arr = [0] * (k + 4)
                if i < 2 * k:
                    arr[i % k] = arr[k + (i < k)] = 1
                else:
                    arr[i - 2 * k + k + 2] = 1
                return arr

            self.embedding_e.weight.data = torch.tensor([get_row(i) for i in range(2 * k + 2)])
            self.embedding_e.weight.requires_grad = False
            self.embedding.to(self.device)
            self.embedding_e.to(self.device)

        if self.e_type == 'same_trained':
            same_emb = torch.zeros((1, self.input_size))
            position_embeddings.weight.data = same_emb.repeat((MAX_LEN, 1)).to(self.device)
            position_embeddings.weight.requires_grad = True

        if self.e_type == 'none':
            position_embeddings.weight.data.zero_()
            position_embeddings.weight.requires_grad = False

        assert not (self.token_embedding_type == 'one_hot' and self.zero_init_emb_dec)
        if self.token_embedding_type == 'one_hot':
            assert self.input_size >= self.vocab_size, 'each token should be able to have different embedding'
            one_hot_embeddings = torch.zeros((self.vocab_size, self.hidden_size))
            for i in range(self.vocab_size):
                one_hot_embeddings[i][i] = 1.0
            self.set_embeddings(one_hot_embeddings, freeze=True)
        if self.zero_init_emb_dec:
            zero_embeddings = self.zero_init_noise * torch.rand((self.vocab_size, self.hidden_size))
            self.set_embeddings(zero_embeddings, freeze=False)

    def set_embeddings(self, embeddings, freeze):
        if self.lm_type in {'BertForMaskedLM', 'BertForMaskedLMCustom'}:
            self.model.bert.embeddings.word_embeddings = nn.Embedding.from_pretrained(
                embeddings, freeze=freeze
            ).to(self.device)
        else:
            raise NotImplementedError('Model not supported.')

    def component_forward(self, batch, attention_mask):
        """ Computes the forward pass to construct prefix representations.
        Arguments:
          batch: (batch_len, seq_len) vectors representing
                 contexts
        Returns:
          hiddens: (batch_len, seq_len, hidden_size)
                   recurrent state vectors for each token in input.
        """
        if self.e_type in ['default', 'cos', 'same_trained', 'none']:
            return self.model.forward(batch, attention_mask=attention_mask).values()
        else:
            vec1 = self.embedding(batch)
            pos = torch.ones(batch.size(), device=self.device).cumsum(-1) - 1
            vec2 = self.embedding_p(pos.long())
            if self.e_type == 'p':
                vec = torch.cat((vec1, vec2), -1)
            else:
                vec3 = self.embedding_e(batch)
                vec = torch.cat((vec1, vec2, vec3), -1)
            return self.model.forward(inputs_embeds=vec, attention_mask=attention_mask).values()

    def forward(self, batch, attention_mask, label_batch=None):
        assert label_batch is None
        return self.component_forward(batch, attention_mask)
