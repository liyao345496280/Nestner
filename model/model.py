from torch import nn
from transformers import AutoModel
from fastNLP import seq_len_to_mask
from torch_scatter import scatter_max
import torch
import torch.nn.functional as F
from .cnn import MaskCNN
from .multi_head_biaffine import MultiHeadBiaffine
from .MCattention import MCattention

class Ner(nn.Module):
    def __init__(self, model_name, num_ner_tag, cnn_dim=200, biaffine_size=200,
                 size_embed_dim=0, logit_drop=0, kernel_size=3, n_head=4, cnn_depth=3):
        super(Ner, self).__init__()
        self.pretrain_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.pretrain_model.config.hidden_size

        if size_embed_dim!=0:
            n_pos = 30
            self.size_embedding = torch.nn.Embedding(n_pos, size_embed_dim)
            _span_size_ids = torch.arange(512) - torch.arange(512).unsqueeze(-1)
            _span_size_ids.masked_fill_(_span_size_ids < -n_pos/2, -n_pos/2)
            _span_size_ids = _span_size_ids.masked_fill(_span_size_ids >= n_pos/2, n_pos/2-1) + n_pos/2
            self.register_buffer('span_size_ids', _span_size_ids.long())
            hsz = biaffine_size*2 + size_embed_dim + 2
        else:
            hsz = biaffine_size*2+2
        biaffine_input_size = hidden_size

        self.head_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(biaffine_input_size, biaffine_size),
            nn.LeakyReLU(),
        )
        self.tail_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(biaffine_input_size, biaffine_size),
            nn.LeakyReLU(),
        )

        self.dropout = nn.Dropout(0.4)
        if n_head>0:
            self.multi_head_biaffine = MultiHeadBiaffine(biaffine_size, cnn_dim, n_head=n_head)
        else:
            self.U = nn.Parameter(torch.randn(cnn_dim, biaffine_size, biaffine_size))
            torch.nn.init.xavier_normal_(self.U.data)
        self.W = torch.nn.Parameter(torch.empty(cnn_dim, hsz))
        torch.nn.init.xavier_normal_(self.W.data)
        # if cnn_depth>0:
        #     self.cnn = MaskCNN(cnn_dim, cnn_dim, kernel_size=kernel_size, depth=cnn_depth)
        #     # pass
        self.down_fc = nn.Linear(cnn_dim, num_ner_tag)
        self.logit_drop = logit_drop
        self.relation_matrix = nn.Linear(hidden_size ,cnn_dim)
        self.projection_matrix = nn.Linear(hidden_size * 2, hidden_size)
        self.activation = nn.ReLU()
        self.dropout_2 = nn.Dropout(0.1)

        self.rope = RoPE(biaffine_input_size, biaffine_size)

        self.MCattention = MCattention(cnn_dim)


    def forward(self, input_ids, bpe_len, indexes, matrix):
        attention_mask = seq_len_to_mask(bpe_len)  # bsz x length x length
        outputs = self.pretrain_model(input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_states = outputs['last_hidden_state']
        state = scatter_max(last_hidden_states, index=indexes, dim=1)[0][:, 1:]  # bsz x word_len x hidden_size
        lengths, _ = indexes.max(dim=-1)

        head_state = self.head_mlp(state)
        tail_state = self.tail_mlp(state)
        head_state_p, tail_state_p = self.rope(state)
        head_state = head_state+head_state_p
        tail_state = tail_state+tail_state_p
        if hasattr(self, 'U'):
            scores1 = torch.einsum('bxi, oij, byj -> boxy', head_state, self.U, tail_state)
        else:
            scores1 = self.multi_head_biaffine(head_state, tail_state)
        head_state = torch.cat([head_state, torch.ones_like(head_state[..., :1])], dim=-1)
        tail_state = torch.cat([tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)
        affined_cat = torch.cat([self.dropout(head_state).unsqueeze(2).expand(-1, -1, tail_state.size(1), -1),
                                 self.dropout(tail_state).unsqueeze(1).expand(-1, head_state.size(1), -1, -1)], dim=-1)

        if hasattr(self, 'size_embedding'):
            size_embedded = self.size_embedding(self.span_size_ids[:state.size(1), :state.size(1)])
            affined_cat = torch.cat([affined_cat,
                                     self.dropout(size_embedded).unsqueeze(0).expand(state.size(0), -1, -1, -1)], dim=-1)

        scores2 = torch.einsum('bmnh,kh->bkmn', affined_cat, self.W)  # bsz x dim x L x L
        scores = scores2+scores1   # bsz x dim x L x L
        # scores_m = self.matrix(state)
        bsz, dim, l, l = scores.shape
        scores_t = self.MCattention(scores)
        scores_t = scores_t.reshape(bsz, dim, l, l)
        scores = scores+scores_t
        # scores = self.matrix(state)
        # if hasattr(self, 'cnn'):
        #     mask = seq_len_to_mask(lengths)  # bsz x length x length
        #     mask = mask[:, None] * mask.unsqueeze(-1)
        #     pad_mask = mask[:, None].eq(0)
        #     u_scores = scores.masked_fill(pad_mask, 0)
        #     if self.logit_drop != 0:
        #         u_scores = F.dropout(u_scores, p=self.logit_drop, training=self.training)
        #     # bsz, num_label, max_len, max_len = u_scores.size()
        #     u_scores = self.cnn(u_scores, pad_mask)
        #     scores = u_scores + scores

        scores = self.down_fc(scores.permute(0, 2, 3, 1))

        assert scores.size(-1) == matrix.size(-1)

        if self.training:
            flat_scores = scores.reshape(-1)
            flat_matrix = matrix.reshape(-1)
            mask = flat_matrix.ne(-100).float().view(input_ids.size(0), -1)
            flat_loss = F.binary_cross_entropy_with_logits(flat_scores, flat_matrix.float(), reduction='none')
            loss = ((flat_loss.view(input_ids.size(0), -1)*mask).sum(dim=-1)).mean()
            return {'loss': loss}

        return {'scores': scores}



class RoPE(nn.Module):
    def __init__(self, hidden_size,num_ner_tag):
        super().__init__()
        self.dense = nn.Linear(hidden_size, num_ner_tag * 2)
        self.device = 'cuda:0'
        self.activate_qw = nn.Sequential(
            # nn.Dropout(0.4),
            nn.Sigmoid()
        )
        self.activate_kw = nn.Sequential(
            # nn.Dropout(0.4),
            nn.Sigmoid()
        )
        self.num_ner_tag = num_ner_tag
    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self,state):
        batch_size = state.size()[0]
        seq_len = state.size()[1]

        outputs = self.dense(state)
        outputs = torch.split(outputs, self.num_ner_tag, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-1)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., 0].unsqueeze(2), outputs[..., 1].unsqueeze(2)

        pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.num_ner_tag)

        # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos
        # logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        # logits = logits.permute(0, 2, 3, 1)
        qw = self.activate_qw(qw)
        kw = self.activate_kw(kw)



        return qw.squeeze(2),kw.squeeze(2)
