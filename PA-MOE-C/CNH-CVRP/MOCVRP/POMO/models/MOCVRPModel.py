import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .MOELayer import MoE

class CVRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes_kv = None
        self.encoded_nodes_q = None
        # shape: (batch, problem+1, EMBEDDING_DIM)
        embedding_dim = self.model_params['embedding_dim']
        hyper_hidden_embd_dim = 256

        self.hyper_fc2 = nn.Linear(embedding_dim, hyper_hidden_embd_dim, bias=True)
        self.hyper_fc3 = nn.Linear(hyper_hidden_embd_dim, embedding_dim, bias=True)
        
        self.aux_loss = 0

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)
        pref = reset_state.preference

        self.encoded_nodes_q, moe_loss, mid_embd_pref = self.encoder(depot_xy, node_xy_demand, pref)
        
        
        self.mid_embd_pref = mid_embd_pref
        self.aux_loss = moe_loss
        
        batch_size, problem_size, _ = node_xy.size()
        embedding_dim = self.model_params['embedding_dim']

        # hyper_embd = self.hyper_fc1(pref)
        encoded_ps = position_encoding_init(batch_size, problem_size, embedding_dim, pref.device)
        EP_embedding = self.hyper_fc2(encoded_ps)
        EP_embed = self.hyper_fc3(EP_embedding)
        self.encoded_nodes_kv = self.encoded_nodes_q + EP_embed
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes_kv)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes_q, state.current_node)
            # shape: (batch, pomo, embedding)
            probs, moe_loss = self.decoder(encoded_last_node, self.mid_embd_pref, state.load, ninf_mask=state.ninf_mask)
            

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None  # value not needed. Can be anything.

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

class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']
        
        self.embedding_pref = nn.Linear(2, embedding_dim)
        
        ###### 
        self.embedding_depot = MoE(input_size=2, input_size_pref=embedding_dim, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                    k=self.model_params['topk'], T=1.0, noisy_gating=True, routing_level=self.model_params['routing_level'],
                                    routing_method=self.model_params['routing_method'], moe_model="Linear")
        self.embedding_node = MoE(input_size=3, input_size_pref=embedding_dim, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                    k=self.model_params['topk'], T=1.0, noisy_gating=True, routing_level=self.model_params['routing_level'],
                                    routing_method=self.model_params['routing_method'], moe_model="Linear")
        
        
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand, pref):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)
        
        
        embedded_pref = self.embedding_pref(pref)
        
        moe_loss = 0 
        
        embedded_depot, loss_depot = self.embedding_depot(depot_xy, embedded_pref)
        moe_loss = moe_loss + loss_depot
        
        embedded_node, loss_node = self.embedding_node(node_xy_demand, embedded_pref)
        moe_loss = moe_loss + loss_node


        out = torch.cat((embedded_depot, embedded_node, embedded_pref[:, None, :]), dim=1)
        # shape: (batch, problem+1, embedding)

        for layer in self.layers:
            out = layer(out)

        return out[:, :-1], moe_loss, embedded_pref
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.Wq2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']
        embed_nodes = input1[:, :-1, :]  # (batch, problem, embedding_dim)
        pref_node = input1[:, -1, :][:, None, :]  # (batch, 1, embedding_dim)

        q1 = reshape_by_heads(self.Wq1(input1), head_num=head_num)
        k1 = reshape_by_heads(self.Wk1(input1), head_num=head_num)
        v1 = reshape_by_heads(self.Wv1(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        q2 = reshape_by_heads(self.Wq2(embed_nodes), head_num=head_num)
        k2 = reshape_by_heads(self.Wk2(pref_node), head_num=head_num)
        v2 = reshape_by_heads(self.Wv2(pref_node), head_num=head_num)

        out_concat = multi_head_attention(q1, k1, v1)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        add_concat = multi_head_attention(q2, k2, v2)
        out_concat[:, :-1] = out_concat[:, :-1] + add_concat

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, problem, embedding)


########################################
# DECODER
########################################

class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.hyper_Wq_last = nn.Linear(embedding_dim+1, head_num * qkv_dim, bias=False)
        self.hyper_Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.hyper_Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.hyper_multi_head_combine = MoE(input_size=head_num * qkv_dim, input_size_pref=embedding_dim, output_size=embedding_dim, num_experts=self.model_params['num_experts'],
                                          k=self.model_params['topk'], T=1.0, noisy_gating=True, routing_level=self.model_params['routing_level'],
                                          routing_method=self.model_params['routing_method'], moe_model="Linear")
        
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.hyper_Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.hyper_Wv(encoded_nodes), head_num=head_num)

        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def forward(self, encoded_last_node, mid_embd_pref, load, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        head_num = self.model_params['head_num']

        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+1)

        q_last = reshape_by_heads(self.hyper_Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        # q = self.q1 + self.q2 + q_last
        # # shape: (batch, head_num, pomo, qkv_dim)
        q = q_last

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

       
        if isinstance(self.hyper_multi_head_combine, MoE):
            mh_atten_out, moe_loss = self.hyper_multi_head_combine(out_concat, mid_embd_pref)
        else:
            mh_atten_out = self.hyper_multi_head_combine(out_concat)


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


class AddAndInstanceNormalization(nn.Module):
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


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.gelu(self.W1(input1)))


def position_encoding_init(batch_szie, n_position, emb_dim, device):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = torch.FloatTensor(np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(50)])).to(device)

    position_enc[1:, 0::2] = torch.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = torch.cos(position_enc[1:, 1::2])  # dim 2i+1

    n_size = n_position // 10
    position_encoding = position_enc[n_size]
    return position_encoding[None, None, :].expand(batch_szie, 1, emb_dim)