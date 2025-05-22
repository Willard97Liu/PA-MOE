# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

import time

__all__ = ['MoE']


class SparseDispatcher(object):
    """
        Helper for implementing a mixture of experts.
        The purpose of this class is to create input minibatches for the
        experts and to combine the results of the experts to form a unified
        output tensor.
        There are two functions:
        dispatch - take an input Tensor and create input Tensors for each expert.
        combine - take output Tensors from each expert and form a combined output
          Tensor.  Outputs from different experts for the same batch element are
          summed together, weighted by the provided "gates".
        The class is initialized with a "gates" Tensor, which specifies which
        batch elements go to which experts, and the weights to use when combining
        the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
        The inputs and outputs are all two-dimensional [batch, depth].
        Caller is responsible for collapsing additional dimensions prior to
        calling this class and reshaping the output to the original shape.
        See common_layers.reshape_like().
        Example use:
        gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
        inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
        experts: a list of length `num_experts` containing sub-networks.
        dispatcher = SparseDispatcher(num_experts, gates)
        expert_inputs = dispatcher.dispatch(inputs)
        expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
        outputs = dispatcher.combine(expert_outputs)
        The preceding code sets the output for a particular example b to:
        output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
        This class takes advantage of sparsity in the gate matrix by including in the
        `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates, routing_level="node"):
        """
            Create a SparseDispatcher.
        """

        self._gates = gates
        self._num_experts = num_experts
        self._routing_level = routing_level

        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)

        # drop indices: corresponds to which expert
        _, self._expert_index = sorted_experts.split(1, dim=1)

        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]

        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()

        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]

        # get the corresponding weights
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """
            Create one input Tensor for each expert.
            The `Tensor` for a expert `i` contains the slices of `inp` corresponding
            to the batch elements `b` where `gates[b, i] > 0`.
            Args:
              inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
            Returns:
              a list of `num_experts` `Tensor`s with shapes
                `[expert_batch_size_i, <extra_input_dims>]`.
        """
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """
            Sum together the expert output, weighted by the gates.
            The slice corresponding to a particular batch element `b` is computed
            as the sum over all experts `i` of the expert output, weighted by the
            corresponding gate values.  If `multiply_by_gates` is set to False, the
            gate values are ignored.
            Args:
              expert_out: a list of `num_experts` `Tensor`s, each with shape
                `[expert_batch_size_i, <extra_output_dims>]`.
              multiply_by_gates: a boolean
            Returns:
              a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        # stitched = torch.cat(expert_out, 0).exp()
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            if self._routing_level == "node":
                stitched = stitched.mul(self._nonzero_gates)
            elif self._routing_level == "instance":
                stitched = stitched.mul(self._nonzero_gates.unsqueeze(1))

        if self._routing_level == "node":
            zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(-1), requires_grad=True, device=stitched.device)
        elif self._routing_level == "instance":
            zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(-1), requires_grad=True, device=stitched.device)

        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        
        combined[combined == 0] = np.finfo(float).eps


        # back to log space
        # return combined.log()
        return combined

    def expert_to_gates(self):
        """
            Gate values corresponding to the examples in the per-expert `Tensor`s.
            Returns:
              a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
                  and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    def __init__(self, input_size=128, output_size=128, hidden_size=512):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))



class MoE(nn.Module):
    """
        Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
        Args:
        input_size: integer - size of the input
        output_size: integer - size of the output
        num_experts: an integer - number of experts
        hidden_size: an integer - hidden size of the experts
        k: an integer - how many experts to use for each batch element
        T: float - temperature to control the entropy of probability distribution
        noisy_gating: boolean - only used for input_choice routing method
        routing_level: string - ["node", "instance", "problem"]
        routing_method: string - ["input_choice", "expert_choice", "soft_moe"]
    """

    def __init__(self, input_size, input_size_pref, output_size, num_experts, hidden_size=None, k=1, T=1.0, noisy_gating=True, routing_level="node", routing_method="input_choice", moe_model="MLP"):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.routing_level = routing_level
        self.routing_method = routing_method
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.input_size_pref = input_size_pref
        self.hidden_size = hidden_size
        self.k = k
        self.T = T

        # instantiate experts and gating/routing network
        if moe_model == "MLP":
            self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for _ in range(self.num_experts)])
        elif moe_model == "Linear":
            self.experts = nn.ModuleList([nn.Linear(self.input_size, self.output_size) for _ in range(self.num_experts)])
        else:
            raise NotImplementedError

        if routing_method == "soft_moe":
            self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts * k), requires_grad=True)
        else:
            self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
            
        # 这里设置对偏好的编码，大小应该是 (torch.zeros(input_size, num_experts)
        self.pref_gate = nn.Parameter(torch.zeros(input_size_pref, num_experts), requires_grad=True)
        
            
            
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        if not (routing_level in ["node", "instance"] and routing_method == "input_choice"):
            # Refer to: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py
            torch.nn.init.kaiming_uniform_(self.w_gate, a=math.sqrt(5))

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(-1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """
            The squared coefficient of variation of a sample.
            Useful as a loss to encourage a positive distribution to be more uniform.
            Epsilons added for numerical stability.
            Returns 0 for an empty Tensor.
            Args:
            x: a `Tensor`.
            Returns:
            a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """
            Compute the "true load per expert", given the gates.
            The load is the number of data/instances for which the corresponding gate is >0.
            Args:
            gates: a `Tensor` of shape [batch_size, n]
            Returns:
            a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """
            Helper function to NoisyTopKGating.
            Computes the probability that value is in top k, given different random noise.

            "This gives us a way of backpropagating from a loss that balances the number
            of times each expert is in the top k experts per example."

            In the case of no noise, pass in None for noise_stddev, and the result will
            not be differentiable.
            Args:
            clean_values: a `Tensor` of shape [batch, n].
            noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
              normally distributed noise with standard deviation noise_stddev.
            noise_stddev: a `Tensor` of shape [batch, n], or None
            noisy_top_values: a `Tensor` of shape [batch, m].
               "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
            Returns:
            a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)  # (batch_size, 1)
        is_in = torch.gt(noisy_values, threshold_if_in)  # (batch_size, num_experts)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)

        return prob

    def noisy_top_k_gating(self, x, pref, train, noise_epsilon=1e-2):
        """
            Noisy top-k gating.
              See paper: https://arxiv.org/abs/1701.06538.
              Args:
                x: input Tensor with shape [batch_size, input_size]
                train: a boolean - we only add noise at training time.
                noise_epsilon: a float
              Returns:
                gates: a Tensor with shape [batch_size, num_experts]
                load: a Tensor with shape [num_experts]
        """
        # for instance-level routing: [batch_size, problem, input_size] -> [batch_size, input_size]
        
        
        x = x.mean(1) if self.routing_level == "instance" else x
        
        clean_logits = x @ self.w_gate + pref @ self.pref_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=-1)
        top_k_logits = top_logits[..., :self.k]
        top_k_indices = top_indices[..., :self.k]
        top_k_gates = self.softmax(top_k_logits / self.T)

        zeros = torch.zeros_like(logits, requires_grad=True)  # (batch_size, num_experts)
        gates = zeros.scatter(-1, top_k_indices, top_k_gates)  # non-topk elements will be 0

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load


    def forward(self, x, pref, loss_coef=1e-3, prob_emb=None):
        
        # the input must have the shape of [batch_size, input_size]
        output_shape = list(x.size()[:-1]) + [self.output_size]
        
        
        
        batch_size = x.size(0)  # 获取 batch_size
        problem_size = x.size(1)  # 获取 problem_size
        
        pref = pref.expand(batch_size, -1)  # 变成 (batch_size, 15)
        
        pref = pref[:, None, :].expand(-1, problem_size, -1)
        
        
        x = x.reshape(-1, self.input_size) if x.dim() != 2 else x
        
        
        
        pref = pref.reshape(-1, self.input_size_pref) if pref.dim() != 2 else pref
        
    
        if self.routing_level in ["node", "instance"]:
                gates, load = self.noisy_top_k_gating(x, pref, self.training)
                # calculate importance loss
                importance = gates.sum(0)
                loss = self.cv_squared(importance) + self.cv_squared(load)
                loss *= loss_coef
                dispatcher = SparseDispatcher(self.num_experts, gates, routing_level=self.routing_level)
                expert_inputs = dispatcher.dispatch(x)
                
                # gates = dispatcher.expert_to_gates()
                expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]  # [(batch_size*, num_experts), ...]
                
                
                y = dispatcher.combine(expert_outputs)
                
                
                
                
                
                # an example of loss scales: reinforce_loss: tensor(-9.3630) aug_loss: tensor(0.0001)
                return y.reshape(output_shape), loss
