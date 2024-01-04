
import os
import math
import sys

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load
from torch.utils.checkpoint import checkpoint
from torch import jit
from typing import Any, Dict, List, Optional, Tuple

module_path = os.path.dirname(__file__)
pcfg_kernel = None

def get_pcfg_kernel():
    global pcfg_kernel
    if not torch.cuda.is_available():
        raise RuntimeError("You need GPU to use the custom cuda operations")
    if pcfg_kernel is not None:
        return pcfg_kernel
    else:
        print("Start compiling cuda operations for PCFG...(It usually takes a few minutes for the first time running.)", file=sys.stderr, flush=True)

        if int(torch.version.cuda.split(".")[0]) < 11:
            extra_include_paths = [os.path.join(module_path, "../../cub")]
        else:
            extra_include_paths = None

        pcfg_kernel = load(
            "pcfg_loss_fn",
            sources=[
                os.path.join(module_path, "pcfg_loss.cpp"),
                os.path.join(module_path, "pcfg_loss.cu"),
                os.path.join(module_path, "pcfg_loss_backward.cu"),
                os.path.join(module_path, "pcfg_best_tree.cu"),
                os.path.join(module_path, "pcfg_viterbi.cu"),
                os.path.join(module_path, "logsoftmax_gather.cu"),
            ],
            extra_cflags=['-DOF_SOFTMAX_USE_FAST_MATH', '-O3'],
            extra_cuda_cflags=['-DOF_SOFTMAX_USE_FAST_MATH', '-O3', '-lineinfo'],
            extra_include_paths=extra_include_paths,
        )
        print("PCFG Cuda operations compiled", file=sys.stderr, flush=True)
        return pcfg_kernel

class CUDAPCFGLossFunc(Function):
    config = 1
    config1 = 1
    config2 = 1

    @staticmethod
    def forward(
        ctx,
        match_all, # bsz * tarlen * prelen
        links, # bsz * prelen * translen
        output_length, # bsz
        target_length, # bsz
    ):
       
        batch_size, prelen, tarlen = match_all.shape
        _, _, max_left, _ = links.shape
        max_left = max_left-1

        require_gradient = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        match_all = match_all.contiguous()
        links = links.contiguous()
        S, C = get_pcfg_kernel().pcfg_loss(match_all, links, output_length, target_length, CUDAPCFGLossFunc.config) # bsz * prelen * tarlen
        if require_gradient:
            match_all_grad, links_grad = get_pcfg_kernel().pcfg_loss_backward(torch.ones((batch_size)).to(C), S, C, match_all, links, output_length, target_length, CUDAPCFGLossFunc.config)
            ctx.save_for_backward(match_all_grad, links_grad)
        return S[range(batch_size), 0, 0]
        
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            match_all_grad, links_grad = ctx.saved_tensors
            batch_size, _, _ = match_all_grad.shape
            return match_all_grad * grad_output.view(batch_size, 1, 1), links_grad * grad_output.view(batch_size, 1, 1, 1), None, None
        else:
            return None, None, None, None

cuda_pcfg_loss = CUDAPCFGLossFunc.apply



class CUDAPCFGBestTreeFunc(Function):
    config = 1

    @staticmethod
    def forward(
        ctx,
        match_all, # bsz * tarlen * prelen
        links, # bsz * prelen * translen
        output_length, # bsz
        target_length, # bsz
    ):
        
        batch_size, prelen, tarlen = match_all.shape
        _, _, max_left, _ = links.shape
        max_left = max_left-1

        match_all = match_all.contiguous()
        links = links.contiguous()
        S, C = get_pcfg_kernel().pcfg_loss(match_all, links, output_length, target_length, CUDAPCFGBestTreeFunc.config) # bsz * prelen * tarlen

        tree = get_pcfg_kernel().pcfg_best_tree(S, C, match_all, links, output_length, target_length, CUDAPCFGBestTreeFunc.config) # bsz * prelen * tarlen
        
        return tree, S[range(batch_size), 0, 0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None

cuda_pcfg_best_tree = CUDAPCFGBestTreeFunc.apply

class CUDAPCFGViterbiFunc(Function):
    config = 1

    @staticmethod
    def forward(
        ctx,
        ob_lprob, # bsz * tarlen * prelen
        links, # bsz * prelen * translen
        output_length, # bsz
    ):

        batch_size = ob_lprob.size(0)
        ob_lprob = ob_lprob.contiguous()
        links = links.contiguous()
        S, R, L, M = get_pcfg_kernel().pcfg_viterbi(ob_lprob, links, output_length, CUDAPCFGBestTreeFunc.config) # bsz * prelen * tarlen
        # print(S[range(batch_size), 0, :])
        # assert(False)
        return S[range(batch_size), 1, :], R, L, M
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None

cuda_pcfg_viterbi = CUDAPCFGViterbiFunc.apply

class DagLogsoftmaxGatherFunc(Function):

    @staticmethod
    def forward(
        ctx,
        word_ins_out, # bsz * prelen * vocabsize
        select_idx # bsz * prelen * slen
    ):
       
        require_gradient = ctx.needs_input_grad[0]
        selected_result = get_pcfg_kernel().logsoftmax_gather(word_ins_out, select_idx, require_gradient)
        # Note: the cuda kernel will modify word_ins_out and then reuse it in backward
        ctx.mark_dirty(word_ins_out)
        ctx.set_materialize_grads(False)

        if require_gradient:
            ctx.save_for_backward(word_ins_out, select_idx)
            ctx.has_backward = False
        return word_ins_out, selected_result # bsz * prelen * slen

    @staticmethod
    def backward(ctx, grad_word_ins_out, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None
        assert grad_word_ins_out is None, "Cannot reuse word_ins_out after logsoftmax_gather"
        if grad_output is None:
            return None, None

        assert not ctx.has_backward, "Cannot backward twice in logsoftmax_gather"
        ctx.has_backward = True

        grad_input, selected_idx = ctx.saved_tensors
        grad_input.mul_(grad_output.sum(-1, keepdim=True).neg_().to(grad_input.dtype))
        grad_input.scatter_add_(-1, selected_idx, grad_output.to(grad_input.dtype))

        return grad_input, None

dag_logsoftmax_gather_inplace = DagLogsoftmaxGatherFunc.apply

def viterbi_decoding(pred_length, output_length, L_trace, R_trace, M_trace, unreduced_tokens, left_tree_mask, pad_index):
    return get_pcfg_kernel().viterbi_decoding(pred_length, output_length, L_trace, R_trace, M_trace, unreduced_tokens, left_tree_mask, pad_index)

if __name__ == "__main__":
    get_pcfg_kernel()
