import torch
import torch.nn as nn
import torch.nn.functional as F

from .MOELayer import MoE

class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem, EMBEDDING_DIM)
        self.aux_loss = 0

    def pre_forward(self, reset_state, pref):
        self.encoded_nodes, moe_loss, embedded_pref = self.encoder(reset_state.problems, pref)
        # shape: (batch, problem, EMBEDDING_DIM)
        
        self.aux_loss = moe_loss
        self.embedded_pref = embedded_pref
        
        
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        
        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs, moe_loss = self.decoder(encoded_last_node, self.embedded_pref, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)
            self.aux_loss  = self.aux_loss + moe_loss

            if self.training or self.model_params['eval_type'] == 'softmax':
                selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                    .squeeze(dim=1).reshape(batch_size, pomo_size)
                # shape: (batch, pomo)

                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                    .reshape(batch_size, pomo_size)
                # shape: (batch, pomo)

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None


        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_pref = nn.Linear(3, embedding_dim)
        
        self.embedding = MoE(input_size=6, input_size_pref=embedding_dim, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                    k=self.model_params['topk'], T=1.0, noisy_gating=True, routing_level=self.model_params['routing_level'],
                                    routing_method=self.model_params['routing_method'], moe_model="Linear")
        
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data, pref):
        # data.shape: (batch, problem, 2)

        embedded_pref = self.embedding_pref(pref)[None, None, :].repeat(data.shape[0], 1, 1)
        
        moe_loss = 0 
        embedded_input, loss_data = self.embedding(data, embedded_pref)
        moe_loss = moe_loss + loss_data
        
        out = torch.cat((embedded_input, embedded_pref), dim=1)
        # out = embedded_input
        for layer in self.layers:
            out, loss = layer(out, embedded_pref)
            moe_loss = moe_loss + loss

        return out, moe_loss, embedded_pref

class FiLM(nn.Module):
    """ Feature-wise Linear Modulation (FiLM) layer"""

    def __init__(self, input_size, output_size, num_film_layers=1, layer_norm=False):
        """
        :param input_size: feature size of x_cond
        :param output_size: feature size of x_to_film
        :param layer_norm: true or false
        """
        super(FiLM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_film_layers = num_film_layers
        self.layer_norm = nn.LayerNorm(output_size) if layer_norm else None
        film_output_size = self.output_size * num_film_layers * 2
        self.gb_weights = nn.Linear(self.input_size, film_output_size, bias=False)
        # self.gb_weights.bias.data.fill_(0)

    def forward(self, x_cond, x_to_film):
        gb = self.gb_weights(x_cond)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        out = gamma * x_to_film + beta
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        
        self.feedForward = MoE(input_size=embedding_dim, input_size_pref=embedding_dim, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                   hidden_size=self.model_params['ff_hidden_dim'], k=self.model_params['topk'], T=1.0, noisy_gating=True,
                                   routing_level=self.model_params['routing_level'], routing_method=self.model_params['routing_method'], moe_model="MLP")
        
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

        self.node_conditional_alignment = FiLM(embedding_dim, embedding_dim)  # FiLM

    def forward(self, input1, mid_embd_pref):
        input = input1.clone()
        input[:, :-1] = self.node_conditional_alignment(x_cond=input1[:, -1, None].repeat(1, input1.shape[1] - 1, 1), x_to_film=input1[:, :-1])

        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input), head_num=head_num)
        k = reshape_by_heads(self.Wk(input), head_num=head_num)
        v = reshape_by_heads(self.Wv(input), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        # out2 = self.feedForward(out1)
        out2, moe_loss = self.feedForward(out1, mid_embd_pref)
        
        out3 = self.addAndNormalization2(out1, out2)

        return out3, moe_loss
        # shape: (batch, problem, EMBEDDING_DIM)


########################################
# DECODER
########################################

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        # self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        
        self.multi_head_combine = MoE(input_size=head_num * qkv_dim, input_size_pref=embedding_dim, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                   hidden_size=self.model_params['ff_hidden_dim'], k=self.model_params['topk'], T=1.0, noisy_gating=True,
                                   routing_level=self.model_params['routing_level'], routing_method=self.model_params['routing_method'], moe_model="MLP")
    
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention
        
    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        
        self.single_head_key = encoded_nodes[:, :-1].transpose(1, 2)
     
    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']

        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, mid_embd_pref, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
       
        q = self.q_first + q_last 
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=torch.cat(
            (ninf_mask, torch.zeros(ninf_mask.shape[0], ninf_mask.shape[1], 1)), dim=-1))
        # out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        # mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)
        mh_atten_out, moe_loss = self.multi_head_combine(out_concat, mid_embd_pref)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs, moe_loss


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
