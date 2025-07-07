import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union


import torch.cuda.nvtx as nvtx


# torch._dynamo.config.capture_dynamic_output_shape_ops = True

# --- Triton Kernel ---
try:
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_D': 64}, num_warps=2),
            triton.Config({'BLOCK_SIZE_D': 128}, num_warps=4),
            triton.Config({'BLOCK_SIZE_D': 256}, num_warps=4),
            triton.Config({'BLOCK_SIZE_D': 512}, num_warps=8),
        ],
        key=['D', 'B_SIZE'],
    )
    @triton.jit
    def router_forward_kernel(
        # --- Pointers to Tensors ---
        X_ptr,
        Stacked_Gate_Weights_ptr,
        Stacked_Up_Weights_ptr,
        Expert_IDs_ptr,
        Out_Scores_ptr,
        # --- Tensor dimensions ---
        D, E_out, E_in, B_SIZE,
        # --- Strides ---
        stride_x_bk, stride_x_d,
        stride_gw_eout, stride_gw_ein_bsize, stride_gw_d,
        stride_uw_eout, stride_uw_ein_bsize, stride_uw_d,
        stride_eid_bk,
        stride_os_bk, stride_os_ein,
        # --- Meta-parameters ---
        BLOCK_SIZE_D: tl.constexpr, # Block size for D dimension
    ):
        """
        Triton Kernel to fuse the inner expert scoring logic.
        Grid: (B*K, E_in)
        """
        # --- Get program IDs to identify the current instance ---
        pid_token = tl.program_id(axis=0)  # Current token index (0 to B*K-1)
        pid_inner_expert = tl.program_id(axis=1)  # Current inner expert index (0 to E_in-1)

        # --- Load the outer expert ID for the current token ---
        # Expert_IDs_ptr is of shape (B*K), so we just need pid_token
        expert_id_ptr = Expert_IDs_ptr + pid_token * stride_eid_bk
        outer_expert_id = tl.load(expert_id_ptr)

        # --- Initialize accumulator for the mean score ---
        total_score_acc = 0.0

        # --- Loop over the 'bigger_size' dimension ---
        # Each inner expert has B_SIZE sub-vectors
        for b_idx in range(B_SIZE):
            # --- Calculate dot product for Gate and Up weights ---
            gate_acc = tl.zeros((), dtype=tl.float32)
            up_acc = tl.zeros((), dtype=tl.float32)

            # Loop over the hidden dimension D in blocks
            for d_start in range(0, tl.cdiv(D, BLOCK_SIZE_D)):
                d_offsets = d_start * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
                d_mask = d_offsets < D

                # 1. Load a block of the input vector x
                x_ptr = X_ptr + pid_token * stride_x_bk + d_offsets * stride_x_d
                x_block = tl.load(x_ptr, mask=d_mask, other=0.0)

                # 2. Calculate offset for the current weight row
                # Weight row index = inner_expert_idx * B_SIZE + b_idx
                weight_row_offset = (pid_inner_expert * B_SIZE + b_idx) * stride_gw_ein_bsize

                # 3. Load a block of the gate weight vector
                gate_weight_ptr = (Stacked_Gate_Weights_ptr +
                                   outer_expert_id * stride_gw_eout +
                                   weight_row_offset +
                                   d_offsets * stride_gw_d)
                gate_weight_block = tl.load(gate_weight_ptr, mask=d_mask, other=0.0)

                # 4. Load a block of the up weight vector
                up_weight_ptr = (Stacked_Up_Weights_ptr +
                                 outer_expert_id * stride_uw_eout +
                                 weight_row_offset + # same offset logic
                                 d_offsets * stride_uw_d)
                up_weight_block = tl.load(up_weight_ptr, mask=d_mask, other=0.0)

                # 5. Accumulate dot product
                gate_acc += tl.sum(x_block * gate_weight_block)
                up_acc += tl.sum(x_block * up_weight_block)

            # --- Apply activation and calculate score for this sub-vector ---
            # silu(x) = x * sigmoid(x)
            silu_gate = gate_acc * tl.sigmoid(gate_acc)
            sub_score = tl.abs(silu_gate * up_acc)

            # Accumulate the score
            total_score_acc += sub_score

        # --- Calculate the mean score ---
        mean_score = total_score_acc / B_SIZE

        # --- Write the final score to the output tensor ---
        out_ptr = Out_Scores_ptr + pid_token * stride_os_bk + pid_inner_expert * stride_os_ein
        tl.store(out_ptr, mean_score)

    def _router_forward(
        x: torch.Tensor,
        gate_weights: torch.Tensor,
        up_weights: torch.Tensor,
        expert_ids: torch.Tensor,
        inner_num: int,
        bigger_size: int,
    ) -> torch.Tensor:
        """Wrapper function to launch the Triton kernel."""
        # --- Shape checks and setup ---
        assert x.is_cuda and gate_weights.is_cuda and up_weights.is_cuda and expert_ids.is_cuda
        assert x.is_contiguous() and gate_weights.is_contiguous() and up_weights.is_contiguous()

        B_K, D = x.shape
        E_out, E_in_B_size, _ = gate_weights.shape
        E_in = inner_num
        B_SIZE = bigger_size
        assert E_in_B_size == E_in * B_SIZE

        # --- Output tensor ---
        # Shape: (B*K, E_in)
        out_scores = torch.empty((B_K, E_in), device=x.device, dtype=torch.float32)

        # --- Grid definition ---
        grid = (B_K, E_in)

        # --- for debug ---
        # nvtx_tag = (
        #     f"RouterForward: B*K={B_K}, D={D}, E_out={E_out}, "
        #     f"E_in={E_in}, B_SIZE={B_SIZE}"
        # )
        # nvtx.range_push(nvtx_tag)

        # --- Kernel launch ---
        router_forward_kernel[grid](
            x, gate_weights, up_weights, expert_ids, out_scores,
            D, E_out, E_in, B_SIZE,
            x.stride(0), x.stride(1),
            gate_weights.stride(0), gate_weights.stride(1), gate_weights.stride(2),
            up_weights.stride(0), up_weights.stride(1), up_weights.stride(2),
            expert_ids.stride(0),
            out_scores.stride(0), out_scores.stride(1),
        )

        # --- for debug ---
        # nvtx.range_pop()

        return out_scores

except ImportError:
    print("Triton not found. Falling back to PyTorch implementation.")
    _router_forward = None

class RouterCompoundFast(nn.Module):
    def __init__(self, config, prefix="gate"):
        super().__init__()
        self.norm_topk_prob: bool = True
        self.n_routed_experts = config.n_routed_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.out_gate_weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim)),requires_grad=False
        )
        self.inner_num = config.inner_num
        self.acti_num = config.num_experts_per_tok
        self.acti_pattern = config.acti_pattern
        self.routed_scaling_factor = config.routed_scaling_factor
        
        assert len(self.acti_pattern) == self.acti_num 

        self.hidden_size = config.hidden_size
        self.bigger_size = config.bigger_size
                
        self.stacked_in_gate_weights = nn.Parameter(
            torch.empty((self.n_routed_experts, self.bigger_size*self.inner_num, self.gating_dim)),requires_grad=False
        )
        self.stacked_in_up_weights = nn.Parameter(
            torch.empty((self.n_routed_experts, self.bigger_size*self.inner_num, self.gating_dim)),requires_grad=False
        )
        
        self.deepseek_style = self.deepseek_style = True if 'deepseek' in config.model_type else False
        self.prefix = prefix

    def init_weight(self, out_gate: nn.Linear, in_gates_list: nn.ModuleList,):
        self.out_gate_weight = nn.Parameter(out_gate.weight.data)
        
        stacked_gate_weights = torch.stack([g.gate.weight for g in in_gates_list])
        stacked_up_weights = torch.stack([g.up.weight for g in in_gates_list])

        self.stacked_in_gate_weights = nn.Parameter(stacked_gate_weights)
        self.stacked_in_up_weights = nn.Parameter(stacked_up_weights)

    def forward_torch(self, x: torch.Tensor, flat_x: torch.Tensor, flat_expert_ids: torch.Tensor) -> torch.Tensor:
        """Original PyTorch implementation for Step 3 for comparison or fallback."""
        batch_gate_weights = self.stacked_in_gate_weights[flat_expert_ids]
        batch_up_weights = self.stacked_in_up_weights[flat_expert_ids]
        
        flat_x_reshaped = flat_x.unsqueeze(-1)
        
        gate_out = torch.bmm(batch_gate_weights, flat_x_reshaped).squeeze(-1)
        up_out = torch.bmm(batch_up_weights, flat_x_reshaped).squeeze(-1)
        
        scores = (up_out * F.silu(gate_out)).abs()
        scores = scores.view(-1, self.inner_num, self.bigger_size)
        all_inner_scores = scores.mean(dim=2)
        return all_inner_scores

    def forward_in(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: 优化--acti_num有0的时候直接跳

        bs, dim = x.shape
        device = x.device

        # Step 1: Outer expert selection
        logits = F.linear(
            x.type(torch.float32), self.out_gate_weight.type(torch.float32), None
        )
        out_scores = F.softmax(logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(out_scores, self.acti_num, dim=-1)

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        else:
            routing_weights *= self.routed_scaling_factor

        # Step 2: Prepare inputs
        flat_x = x.repeat_interleave(self.acti_num, dim=0)
        flat_expert_ids = selected_experts.reshape(-1)
        flat_weights = routing_weights.reshape(-1)

        # --- Step 3: Inner Expert Scoring (Triton or PyTorch) ---
        if _router_forward is not None and x.is_cuda:
            # 使用 Triton Fused Kernel
            all_inner_scores = _router_forward(
                flat_x,
                self.stacked_in_gate_weights,
                self.stacked_in_up_weights,
                flat_expert_ids,
                self.inner_num,
                self.bigger_size
            )
        else:
            # 使用原始 PyTorch 实现作为后备
            all_inner_scores = self.forward_torch(x, flat_x, flat_expert_ids)
        # all_inner_scores shape: (B*K, E_in)

        # Step 4: Inner top-k routing
        inner_topks = torch.tensor(self.acti_pattern, device=device)[None, :].expand(bs, -1).reshape(-1)
        max_topk = max(self.acti_pattern)
        _, topk_indices = torch.topk(all_inner_scores, k=max_topk, dim=-1)

        arange = torch.arange(max_topk, device=device)[None, :]
        mask = arange < inner_topks[:, None]

        total_activated_experts = sum(self.acti_pattern)

        flat_expert_ids_expanded = flat_expert_ids[:, None].expand(-1, max_topk)
        selected_inner_ids = flat_expert_ids_expanded * self.inner_num + topk_indices

        # final_ids = torch.masked_select(selected_inner_ids, mask).view(bs, total_activated_experts)

        expanded_weights = flat_weights[:, None].expand(-1, max_topk)

        # final_weights = torch.masked_select(expanded_weights, mask).view(bs, total_activated_experts)
        
        # Use static graph for compile. (-10000 is a temporary solution, while torch.inf cannot be used here)
        # P.S. torch.masked_select will lead to a dynamic graph.
        selected_inner_ids_masked = torch.where(mask, selected_inner_ids, torch.full_like(selected_inner_ids, -10000)).view(bs,-1)
        final_ids,_ = torch.topk(selected_inner_ids_masked,k=total_activated_experts,dim=-1)

        masked_weights = torch.where(mask, expanded_weights, torch.full_like(expanded_weights,-10000)).view(bs,-1)
        final_weights,_ = torch.topk(masked_weights,k=total_activated_experts,dim=-1)

        if self.deepseek_style:
            return final_ids, final_weights, None
        else:
            return final_weights, final_ids

    @torch.no_grad()
    def forward(self,
                    hidden_states,
                    gating_output,
                    topk,
                    renormalize):
        return self.forward_in(hidden_states)