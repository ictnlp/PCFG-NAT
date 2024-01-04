import math
import re
import logging
from functools import reduce
import numpy as np
from typing import Union, Tuple, Optional
import sys

import torch
from torch import Tensor
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch.autograd import Function
from ..custom_ops import cuda_pcfg_loss, cuda_pcfg_best_tree
from ..custom_ops import dag_logsoftmax_gather_inplace

from .utilities import parse_anneal_argument, get_anneal_value

logger = logging.getLogger(__name__)

########### gpu use tracker ###########
# import inspect
SHOW_MEMORY_USE=False
if SHOW_MEMORY_USE:
    from fairseq.gpu_mem_track import MemTracker
    gpu_tracker = MemTracker()
########################################

@register_criterion("nat_pcfg_loss")
class NATPCFGoss(FairseqCriterion):

    def __init__(self, cfg, task):
        super().__init__(task)
        self.cfg = cfg
        assert cfg.label_smoothing == 0, "pcfg does not support label smoothing"
        self.glance_strategy = cfg.glance_strategy
        self._glat_p_anneal_params = parse_anneal_argument(cfg.glat_p)

        self.set_update_num(0)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument("--label-smoothing", type=float, default=0, help="DA-Transformer does not use label smoothing for now")
        parser.add_argument("--glat-p", type=str, default="0", help="Glancing probability. 0.5:0.1@200k indicates annealing p from 0.5 to 0.1 in 200k steps.")
        parser.add_argument("--glance-strategy", type=str, default=None, help='Glancing strategy. Possible values: "number-random" or "None" or "CMLM"')
        parser.add_argument("--no-force-emit", action="store_true", help="If true, do not fix the position of glance tokens in the second forward pass")

        parser.add_argument("--torch-pfcg-logsoftmax-gather", action="store_true", help="Use torch implementation for logsoftmax-gather, which supports GPU and CPU device. (Cuda implementation only supports GPU)")
        parser.add_argument("--torch-pfcg-best-alignment", action="store_true", help="Use torch implementation for pfcg-best-alignment, which supports GPU and CPU device. (Cuda implementation only supports GPU)")

    def _compute_loss(self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = utils.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                        nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss_nofactor = loss
        loss = loss * factor

        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor, "ntokens": outputs.shape[0], "loss_nofactor": loss_nofactor}

    def _compute_pfcg_loss(self, outputs, output_masks, targets, target_masks, links, label_smoothing=0.0, name="loss",
                factor=1.0, matchmask=None, keep_word_mask=None, model=None):

        batch_size = outputs.shape[0]
        prelen = outputs.shape[1]
        tarlen = targets.shape[1]

        output_length = output_masks.sum(dim=-1)
        target_length = target_masks.sum(dim=-1)
        

        outputs, match_all = dag_logsoftmax_gather_inplace(outputs, targets.unsqueeze(1).expand(-1, prelen, -1))
        
        nvalidtokens = output_masks.sum()
        eos_mask = torch.zeros_like(match_all).to(match_all).bool()
        eos_mask[range(batch_size), :, target_length-1] = True
        eos_mask[range(batch_size), output_length-1, target_length-1] = False
        match_all_eos_masked = match_all.masked_fill(eos_mask, float("-inf")) 
        if matchmask is not None and not self.cfg.no_force_emit:
            glat_prev_mask = keep_word_mask.unsqueeze(-1)
            
            matchmask = matchmask.transpose(1,2)
            # print(match_all_eos_masked.size(), matchmask.size(), glat_prev_mask.size())
            match_all_eos_masked = match_all_eos_masked.masked_fill(glat_prev_mask, 0) + match_all_eos_masked.masked_fill(~matchmask, float("-inf")).masked_fill(~glat_prev_mask, 0).detach()
        loss_result = cuda_pcfg_loss(match_all_eos_masked, links, output_length, target_length)
        invalid_masks = loss_result.isinf().logical_or(loss_result.isnan())
        # if loss_result.isinf().any():
        #     print(loss_result, match_all_eos_masked.size())
        #     np.savez('/data/guishangtong/PCFG-NAT/fs_plugins/numba_tests/test.npz', match_all=match_all_eos_masked.detach().cpu().numpy(), links=links.detach().cpu().numpy(), output_length=output_length.detach().cpu().numpy(), target_length=target_length.detach().cpu().numpy())
        #     assert(False)
        # print(loss_result)
        # if loss_result.isinf().any():
        #     print(loss_result)
        #     assert(False)
        loss_result.masked_fill_(invalid_masks, 0)
        invalid_nsentences = invalid_masks.sum().detach()

        loss = -(loss_result / target_length).mean()
        
        nll_loss = loss.detach()
        nsentences, ntokens = targets.shape[0], targets.ne(self.task.tgt_dict.pad()).sum()

        loss_nofactor = loss
        loss = loss * factor

        return {"name": name, "loss": loss, "nll_loss": nll_loss,
                "factor": factor, "ntokens": ntokens, "nvalidtokens": nvalidtokens, "nsentences": nsentences,
                "loss_nofactor": loss_nofactor, "invalid_nsentences": invalid_nsentences}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def set_update_num(self, update_num):
        self.glat_p = get_anneal_value(self._glat_p_anneal_params, update_num)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # import gc
        # gc.collect()
        # if SHOW_MEMORY_USE:
        #     print(torch.cuda.memory_reserved() / 1024 / 1024, file=sys.stderr, flush=True)
        #     gpu_tracker.clear_cache()
        if SHOW_MEMORY_USE:
            gpu_tracker.track()

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens = sample["target"]

        # if SHOW_MEMORY_USE:
        #     print(sample["net_input"]["src_tokens"].shape[0], sample["net_input"]["src_tokens"].shape[1], tgt_tokens.shape[1], file=sys.stderr, end=" ")

        if sample.get("update_num", None) is not None: # in training
            self.set_update_num(sample['update_num'])

        prev_output_tokens = model.initialize_output_tokens_by_tokens(src_tokens, tgt_tokens)
        

        if self.glat_p == 0:
            glat = None
        else:
            glat = {
                "context_p": max(self.glat_p, 0),
                "require_glance_grad": False
            }

        def glat_function(model, word_ins_out, tgt_tokens, prev_output_tokens, glat, links=None):
            batch_size, prelen, _, _ = links.shape
            tarlen = tgt_tokens.shape[1]
            nonpad_positions = ~tgt_tokens.eq(model.pad)
            target_length = (nonpad_positions).sum(1)
            output_length = prev_output_tokens.ne(model.pad).sum(1)
            pred_tokens = word_ins_out.argmax(-1)
            word_ins_out, match = dag_logsoftmax_gather_inplace(word_ins_out, tgt_tokens.unsqueeze(1).expand(-1, prelen, -1))
            eos_mask = torch.zeros_like(match).to(match).bool()
            # for batch_id in range(batch_size):
            #     eos_mask[batch_id, :, target_length[batch_id]-1] = True    
            #     eos_mask[batch_id, output_length[batch_id]-1, target_length[batch_id]-1] = False
            eos_mask[range(batch_size), :, target_length-1] = True
            eos_mask[range(batch_size), output_length-1, target_length-1] = False
            match_all_eos_masked = match.masked_fill(eos_mask, float("-inf")) 
            
            
            path, max_tree_lprob = cuda_pcfg_best_tree(match_all_eos_masked, links, output_length, target_length) # batch * prelen
            # if max_tree_lprob.isinf().any():
            #     np.savez('/data/guishangtong/PCFG-NAT/fs_plugins/numba_tests/test.npz', match_all=match_all_eos_masked.detach().cpu().numpy(), links=links.detach().cpu().numpy(), output_length=output_length.detach().cpu().numpy(), target_length=target_length.detach().cpu().numpy())
            #     assert(False)
            invalid_masks = max_tree_lprob.isinf().logical_or(max_tree_lprob.isnan())
            max_tree_lprob.masked_fill_(invalid_masks, 0)
            predict_align_mask = path >= 0
            path[path+1>=(tarlen+1)] = -1
            path[path+1<0] = -1
            # if not ((path+1) < (tarlen+1)).all():
            #     torch.set_printoptions(profile="full")
            #     print(tarlen+1)
            #     print(path[(path+1) >= (tarlen+1)])
            #     print(path+1)
            #     assert(((path+1) < (tarlen+1)).all())
            matchmask = torch.zeros(batch_size, tarlen + 1, prelen, device=match.device, dtype=torch.bool).scatter_(1, path.unsqueeze(1) + 1, 1)[:, 1:]
            
            oracle = tgt_tokens.gather(-1, path.clip(min=0)) # bsz * prelen
            # print(pred_tokens.size(), oracle.size(), matchmask.size())
            
            same_num = ((pred_tokens == oracle) & predict_align_mask).sum(1)

            if self.glance_strategy is None:
                keep_prob = ((target_length - same_num) / target_length * glat['context_p']).unsqueeze(-1) * predict_align_mask.float()

            elif self.glance_strategy in ['number-random']:
                prob = torch.randn(oracle.shape, device=tgt_tokens.device, dtype=torch.float)
                prob.masked_fill_(~predict_align_mask, -100)
                glance_nums = ((target_length - same_num) * glat['context_p'] + 0.5).to(torch.long)
                #prob_thresh = prob.topk(glance_nums.max().clip(min=1))[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh = prob.sort(descending=True)[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh.masked_fill_(glance_nums == 0, 100)
                keep_prob = (prob >= prob_thresh.unsqueeze(-1)).to(prob.dtype)

            elif self.glance_strategy == "cmlm":
                prob = torch.randn(oracle.shape, device=tgt_tokens.device, dtype=torch.float)
                prob.masked_fill_(~predict_align_mask, -100)
                glance_nums = (target_length * torch.rand_like(target_length, dtype=torch.float) + 0.5).to(torch.long)
                #prob_thresh = prob.topk(glance_nums.max().clip(min=1))[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh = prob.sort(descending=True)[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
                prob_thresh.masked_fill_(glance_nums == 0, 100)
                keep_prob = (prob >= prob_thresh.unsqueeze(-1)).to(prob.dtype)

            keep_word_mask = (torch.rand(prev_output_tokens.shape, device=prev_output_tokens.device) < keep_prob).bool()

            glat_prev_output_tokens = prev_output_tokens.masked_fill(keep_word_mask, 0) + oracle.masked_fill(~keep_word_mask, 0)
            output_length = prev_output_tokens.ne(model.pad).sum(dim=-1)
            glat_tgt_tokens = tgt_tokens

            glat_info = {
                "glat_accu": (same_num.sum() / target_length.sum()).detach(),
                "glat_context_p": glat['context_p'],
                "glat_keep": keep_prob.mean().detach(),
                "matchmask": matchmask,
                "keep_word_mask": keep_word_mask,
                "glat_prev_output_tokens": glat_prev_output_tokens,
                "max_tree_lprob": -(max_tree_lprob / target_length).mean(),
            }

            return glat_prev_output_tokens, glat_tgt_tokens, glat_info
        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat, glat_function)

        losses = []

        # PCFG loss
        _losses = self._compute_pfcg_loss(
            outputs["word_ins"].get("out"),
            prev_output_tokens.ne(self.task.tgt_dict.pad()),
            outputs["word_ins"].get("tgt"),
            outputs["word_ins"].get("mask", None),
            outputs["links"],
            name="pcfg-loss",
            factor=1,
            matchmask=outputs.get('matchmask', None),
            keep_word_mask=outputs.get('keep_word_mask', None),
            model=model
        )

        losses += [_losses]
        pcfg_nll_loss = _losses.get("nll_loss", 0.0)
        nsentences = _losses["nsentences"]
        ntokens = _losses["ntokens"]
        nvalidtokens = _losses["nvalidtokens"]
        invalid_nsentences = _losses["invalid_nsentences"]

        #length
        _losses = self._compute_loss(
            outputs["length"].get("out"),
            outputs["length"].get("tgt"),
            None,
            0,
            name="length-loss",
            factor=outputs["length"]["factor"], )
        losses += [_losses]
        length_nll_loss = _losses.get("nll_loss", 0.0)

        loss = sum(l["loss"] for l in losses)

        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "pcfg_nll-loss": pcfg_nll_loss.data,
            "length_nll-loss": length_nll_loss.data,
            "ntokens": ntokens,
            "nvalidtokens": nvalidtokens,
            "nsentences": nsentences,
            "invalid_nsentences": invalid_nsentences,
            "sample_size": sample_size,
            "glat_acc": outputs.get("glat_accu", 0),
            "glat_keep": outputs.get("glat_keep", 0),
            "max_tree_lprob": outputs.get("max_tree_lprob", 0),
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss_nofactor"])
                if reduce
                else l["loss_nofactor"]
            )
        if SHOW_MEMORY_USE:
            gpu_tracker.track()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )  # each batch is 1
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))  # token-level loss

        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nvalidtokens = sum(log.get('nvalidtokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        invalid_nsentences = sum(log.get('invalid_nsentences', 0) for log in logging_outputs)
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))  # token-level loss
        glat_acc = utils.item(sum(log.get("glat_acc", 0) for log in logging_outputs))
        glat_keep = utils.item(sum(log.get("glat_keep", 0) for log in logging_outputs))
        max_tree_lprob = utils.item(sum(log.get("max_tree_lprob", 0) for log in logging_outputs))

        res = {
            "ntokens": utils.item(ntokens),
            "nsentences": utils.item(nsentences),
            "nvalidtokens": utils.item(nvalidtokens),
            "invalid_nsentences": utils.item(invalid_nsentences),
            'tokens_perc': utils.item(nvalidtokens / ntokens),
            'sentences_perc': 1 - utils.item(invalid_nsentences / nsentences),
        }
        res["loss"] = loss / sample_size
        res["glat_acc"] = glat_acc / sample_size
        res["glat_keep"] = glat_keep / sample_size
        res["max_tree_lprob"] = max_tree_lprob / sample_size

        for key, value in res.items():
            metrics.log_scalar(
                key, value, sample_size, round=3
            )

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = utils.item(sum(log.get(key, 0) for log in logging_outputs))
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
