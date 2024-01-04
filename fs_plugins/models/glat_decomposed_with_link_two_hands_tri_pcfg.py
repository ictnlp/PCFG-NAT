from fairseq.models.nat.fairseq_nat_model import FairseqNATModel
import logging
import random
import copy
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn, jit
import torch.nn.functional as F
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.modules import (
    PositionalEmbedding,
)
from .lemon_tree import BinaryTreeNode
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.nat.nonautoregressive_transformer import NATransformerDecoder
from contextlib import contextmanager
from .lemon_tree import *
from ..custom_ops import cuda_pcfg_viterbi, viterbi_decoding
import sys

logger = logging.getLogger(__name__)

@contextmanager
def torch_seed(seed):
    # modified from lunanlp
    state = torch.random.get_rng_state()
    state_cuda = torch.cuda.random.get_rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
        torch.cuda.random.set_rng_state(state_cuda)

# @jit.script
def logsumexp(x: Tensor, dim: int) -> Tensor:
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float('inf'))

@register_model("glat_decomposed_with_link_two_hands_tri_pcfg")
class GlatDecomposedLinkPCFG(FairseqNATModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.init_beam_search()
        self._left_tree_mask = torch.empty(0)
        self._right_tree_mask = torch.empty(0)
        self._main_chain = torch.empty(0)
        self.max_left = 2**self.args.left_tree_layer-1
        self.total_length = 0
        self.total_main_chain = 0
        self.layer_1_count = 0
        self.layer_2_count = 0
        # self.main_chain_subword = [0,0,0]
        # self.left_tree_subword = [0,0,0]
        self.main_chain_subword = 0
        self.left_tree_subword = 0
        self.main_chain_pos_dict = {}
        self.left_tree_pos_dict = {}


    def init_beam_search(self):
        if self.args.decode_strategy == "beamsearch":
            import dag_search
            self.dag_search = dag_search
            dag_search.beam_search_init(self.args.decode_max_batchsize, self.args.decode_beamsize,
                    self.args.decode_top_cand_n, self.decoder.max_positions(), self.tgt_dict, self.args.decode_lm_path)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = GlatLinkDecoderPCFG(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)
        GlatLinkDecoderPCFG.add_args(parser)

        # length prediction
        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            "--pred-length-offset",
            action="store_true",
            help="predicting the length difference between the target and source sentences",
        )
        parser.add_argument(
            "--sg-length-pred",
            action="store_true",
            help="stop the gradients back-propagated from the length predictor",
        )
        parser.add_argument(
            "--length-loss-factor",
            type=float,
            help="weights on the length prediction loss",
        )

        parser.add_argument('--links-feature', type=str, default="feature:position", help="Features used to predict transition.")
        parser.add_argument('--max-transition-length', type=int, default=99999, help="Max transition distance. -1 means no limitation, \
                        which cannot be used for cuda custom operations. To use cuda operations with no limitation, please use a very large number such as 99999.")

        parser.add_argument("--left-tree-layer", type=int, default=None, help="tree layer size of left sub tree")

        parser.add_argument("--src-upsample-scale", type=float, default=None, help="Specify the graph size with a upsample factor (lambda).  Graph Size = \\lambda * src_length")
        parser.add_argument("--src-upsample-fixed", type=int, default=None, help="Specify the graph size by a constant. Cannot use together with src-upsample-scale")
        parser.add_argument("--length-multiplier", type=float, default=None, help="Deprecated") # does not work now
        parser.add_argument('--max-decoder-batch-tokens', type=int, default=None, help="Max tokens for LightSeq Decoder when using --src-upsample-fixed")

        parser.add_argument('--filter-max-length', default=None, type=str, help='Filter the sample that above the max lengths, e.g., "128:256" indicating 128 for source, 256 for target. Default: None, for filtering according max-source-positions and max-target-positions')
        parser.add_argument("--filter-ratio", type=float, default=None, help="Deprecated") # does not work now; need support of trainer.py

        parser.add_argument('--decode-strategy', type=str, default="lookahead", help='One of "greedy", "lookahead", "beamsearch"')

        parser.add_argument('--decode-alpha', type=float, default=1.1, help="Used for length penalty. Beam Search finds the sentence maximize: 1 / |Y|^{alpha} [ log P(Y) + gamma log P_{n-gram}(Y)]")
        parser.add_argument('--decode-beta', type=float, default=1, help="Scale the score of logits. log P(Y, A) := sum P(y_i|a_i) + beta * sum log(a_i|a_{i-1})")
        parser.add_argument('--decode-top-cand-n', type=float, default=5, help="Numbers of top candidates when considering transition")
        parser.add_argument('--decode-gamma', type=float, default=0.1, help="Used for n-gram language model score. Beam Search finds the sentence maximize: 1 / |Y|^{alpha} [ log P(Y) + gamma log P_{n-gram}(Y)]")
        parser.add_argument('--decode-beamsize', type=float, default=100, help="Beam size")
        parser.add_argument('--decode-max-beam-per-length', type=float, default=10, help="Limits the number of beam that has a same length in each step")
        parser.add_argument('--decode-top-p', type=float, default=0.9, help="Max probability of top candidates when considering transition")
        parser.add_argument('--decode-lm-path', type=str, default=None, help="Path to n-gram language model. None for not using n-gram LM")
        parser.add_argument('--decode-max-batchsize', type=int, default=32, help="Should not be smaller than the real batch size (the value is used for memory allocation)")
        parser.add_argument('--decode-dedup', type=bool, default=False, help="Use token deduplication in BeamSearch")



    def extract_links(self, features, prev_output_tokens,
            link_positional, query_linear, key_linear_left, key_linear_right, gate_linear, left_tree_mask, right_tree_mask, main_chain):

        links_feature = vars(self.args).get("links_feature", "feature:position").split(":")

        links_feature_arr = []
        if "feature" in links_feature:
            links_feature_arr.append(features)
        if "position" in links_feature or "sinposition" in links_feature:
            links_feature_arr.append(link_positional(prev_output_tokens))

        features_withpos = torch.cat(links_feature_arr, dim=-1)

        batch_size = features.shape[0]
        seqlen = features.shape[1]
        chunk_num = self.args.decoder_attention_heads
        chunk_size = self.args.decoder_embed_dim // self.args.decoder_attention_heads
        ninf = float("-inf")
        target_dtype = torch.float

        query_chunks = query_linear(features_withpos).reshape(batch_size, seqlen, chunk_num, chunk_size)
        key_chunks_left = key_linear_left(features_withpos).reshape(batch_size, seqlen, chunk_num, chunk_size)
        key_chunks_right = key_linear_right(features_withpos).reshape(batch_size, seqlen, chunk_num, chunk_size)
        log_gates = F.log_softmax(gate_linear(features_withpos), dim=-1, dtype=target_dtype) # batch_size * seqlen * chunk_num
        log_multi_content_ab = (torch.einsum("bicf,bjcf->bijc", query_chunks.to(dtype=target_dtype), key_chunks_left.to(dtype=target_dtype)) / ((chunk_size) ** 0.5))
        log_multi_content_ac = (torch.einsum("bicf,bjcf->bijc", query_chunks.to(dtype=target_dtype), key_chunks_right.to(dtype=target_dtype)) / ((chunk_size) ** 0.5))
        log_multi_content_bc_folded = (torch.einsum("bicf,bjcf->bijc", key_chunks_left.to(dtype=target_dtype), key_chunks_right.to(dtype=target_dtype)) / ((chunk_size) ** 0.5))
        
        link_left_mask = torch.logical_or(prev_output_tokens.eq(self.pad).unsqueeze(1),left_tree_mask.unsqueeze(0))
        link_right_mask = torch.logical_or(prev_output_tokens.eq(self.pad).unsqueeze(1),right_tree_mask.unsqueeze(0))
        
        link_right_mask = ~(~link_right_mask & main_chain.unsqueeze(0).unsqueeze(0))
        output_length = prev_output_tokens.ne(self.pad).sum(dim=-1)
        # assert((output_length - 2))
        link_left_nouse_mask = (~link_left_mask).sum(dim=2, keepdim=True) == 0
        link_right_nouse_mask = (~link_right_mask).sum(dim=2, keepdim=True) == 0

        link_left_mask.masked_fill_(link_left_nouse_mask, False)
        link_right_mask.masked_fill_(link_right_nouse_mask, False)
        link_nouse_mask = torch.logical_or(link_left_nouse_mask, link_right_nouse_mask)
        
        log_multi_content_ab = log_multi_content_ab.masked_fill(link_left_mask.view(batch_size, seqlen, seqlen, 1), ninf)
        log_multi_content_ac = log_multi_content_ac.masked_fill(link_right_mask.view(batch_size, seqlen, seqlen, 1), ninf)

        index = self._max_left_index[:, :seqlen, :,].to(log_multi_content_ab.device)
        log_multi_content_ab = torch.gather(log_multi_content_ab, dim=2, index=index.view(1, seqlen, self.max_left+1, 1).expand(batch_size, -1, -1, chunk_num))

        log_multi_content_bc = log_multi_content_bc_folded.unfold(1, self.max_left, 1)
        
        log_multi_content_bc = torch.roll(log_multi_content_bc, 1, 1)
        log_multi_content_bc = log_multi_content_bc.permute(0,1,4,2,3)
        log_multi_content_bc = torch.cat((log_multi_content_bc_folded[:,0,:,:].view(batch_size, 1, 1, seqlen, chunk_num).expand(-1, seqlen, -1, -1, -1) ,log_multi_content_bc), 2)

        log_multi_content_abc = log_multi_content_ab.unsqueeze(3) + log_multi_content_bc + log_multi_content_ac.unsqueeze(2)

        log_multi_content_abc = F.log_softmax(log_multi_content_abc.reshape(batch_size, seqlen, (self.max_left+1)*seqlen, chunk_num), dim=2)
        
        links = logsumexp(log_multi_content_abc + log_gates.unsqueeze(2), dim=-1)
        links = links.view(batch_size, seqlen, self.max_left+1, seqlen)
        links = links.masked_fill(link_nouse_mask.view(batch_size, seqlen, 1, 1), ninf)

        return links

    def buffered_tree_mask(self, tensor):
        dim = tensor.size(1)
        
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (self._left_tree_mask.size(0) == 0 or self._left_tree_mask.size(0) < dim):
            _left_tree_mask, _right_tree_mask, _main_chain = BinaryTreeNode.get_mask(dim, self.args.left_tree_layer, self.args.src_upsample_scale)                        

            self._left_tree_mask = _left_tree_mask
            self._right_tree_mask = _right_tree_mask
            self._main_chain = _main_chain
            self._max_left_index = torch.arange(0, dim).unsqueeze(0).unsqueeze(-1)
            self._max_left_index = self._max_left_index + torch.zeros(1, dim, self.max_left+1, dtype=torch.int64)
            self._max_left_index[:,:,0] = 0

            for i in range(1, self.max_left+1):
                self._max_left_index[:,:,i] = torch.where((self._max_left_index[:,:,i] - i) < 0, 0, self._max_left_index[:,:,i] - i)
            self._left_tree_mask = self._left_tree_mask.bool()
            self._right_tree_mask = self._right_tree_mask.bool()
            self._main_chain = self._main_chain.bool()

        self._left_tree_mask = self._left_tree_mask.to(tensor.device)
        self._right_tree_mask = self._right_tree_mask.to(tensor.device)
        self._main_chain = self._main_chain.to(tensor.device)
        return self._left_tree_mask[:dim, :dim], self._right_tree_mask[:dim, :dim], self._main_chain[:dim]

    def extract_features(self, prev_output_tokens, encoder_out, rand_seed, require_links=False):
        with torch_seed(rand_seed):
            features, _ = self.decoder.extract_features(
                prev_output_tokens,
                encoder_out=encoder_out,
                embedding_copy=False
            )
            # word_ins_out = self.decoder.output_layer(features)
            left_tree_mask, right_tree_mask, main_chain = self.buffered_tree_mask(features)
            word_ins_out = self.decoder.output_projection(features)
            links = None
            if require_links:
                links = self.extract_links(features, \
                            prev_output_tokens, \
                            self.decoder.link_positional, \
                            self.decoder.query_linear, \
                            self.decoder.key_linear_left, \
                            self.decoder.key_linear_right, \
                            self.decoder.gate_linear, \
                            left_tree_mask, \
                            right_tree_mask, \
                            main_chain, \
                        )

        

        return word_ins_out, links, left_tree_mask, right_tree_mask, main_chain

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat=None, glat_function=None, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )
        rand_seed = random.randint(0, 19260817)
        glat_info = None
        if glat and tgt_tokens is not None:
            with torch.set_grad_enabled(glat.get('require_glance_grad', False)):
                word_ins_out, links, left_tree_mask, right_tree_mask, main_chain = self.extract_features(prev_output_tokens, encoder_out, rand_seed, require_links=True)
                prev_output_tokens, tgt_tokens, glat_info = glat_function(self, word_ins_out, tgt_tokens, prev_output_tokens, glat, links=links)
                word_ins_out = None

        word_ins_out, links, left_tree_mask, right_tree_mask, main_chain = self.extract_features(prev_output_tokens, encoder_out, rand_seed, require_links=True)

        ret = {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "main_chain": main_chain,
                "nll_loss": True,
            }
        }
        ret['links'] = links
        ret['left_tree_mask'] = left_tree_mask
        ret['right_tree_mask'] = right_tree_mask
        ret["length"] = {
            "out": length_out,
            "tgt": length_tgt,
            "factor": self.decoder.length_loss_factor,
        }
        if glat_info is not None:
            ret.update(glat_info)
        return ret


    def initialize_output_tokens_with_length(self, src_tokens, length_tgt):
        max_length = length_tgt.max()
        if length_tgt.min() < 2:
            
            print(length_tgt)
            assert(False)
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)
        return initial_output_tokens

    def initialize_output_tokens_upsample_by_tokens(self, src_tokens):
        if vars(self.args).get("src_upsample_scale", None) is not None:
            length_tgt = torch.sum(src_tokens.ne(self.tgt_dict.pad_index), -1)
            length_tgt = (length_tgt * self.args.src_upsample_scale * (2**self.args.left_tree_layer)).long().clamp_(min=2) + 2
        else:
            length_tgt = torch.zeros(src_tokens.shape[0], device=src_tokens.device, dtype=src_tokens.dtype).fill_(self.args.src_upsample_fixed)
        return self.initialize_output_tokens_with_length(src_tokens, length_tgt)

    def initialize_output_tokens_multiplier_by_tokens(self, src_tokens, tgt_tokens):
        length_tgt = torch.sum(tgt_tokens.ne(self.tgt_dict.pad_index), -1)
        length_tgt = (length_tgt * self.args.length_multiplier).long().clamp_(min=2)
        return self.initialize_output_tokens_with_length(src_tokens, length_tgt)

    def initialize_output_tokens_by_tokens(self, src_tokens, tgt_tokens):
        if vars(self.args).get("src_upsample_scale", None) is not None or vars(self.args).get("src_upsample_fixed", None) is not None:
            return self.initialize_output_tokens_upsample_by_tokens(src_tokens)
        elif vars(self.args).get("length_multiplier", None) is not None:
            return self.initialize_output_tokens_multiplier_by_tokens(src_tokens, tgt_tokens)

    def initialize_output_tokens_upsample(self, encoder_out, src_tokens):
        # length prediction
        if vars(self.args).get("src_upsample_scale", None) is not None:
            length_tgt = torch.sum(src_tokens.ne(self.tgt_dict.pad_index), -1)
            length_tgt = (length_tgt * self.args.src_upsample_scale * (2**self.args.left_tree_layer)).long().clamp_(min=2) + 2
        else:
            length_tgt = torch.zeros(src_tokens.shape[0], device=src_tokens.device, dtype=src_tokens.dtype).fill_(self.args.src_upsample_fixed)
        initial_output_tokens = self.initialize_output_tokens_with_length(src_tokens, length_tgt)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def initialize_output_tokens_multiplier(self, encoder_out, src_tokens):
        # length prediction
        length_tgt = self.decoder.forward_length_prediction(
            self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
            encoder_out=encoder_out,
        )
        length_tgt = (length_tgt * self.args.length_multiplier).long().clamp_(min=2)
        initial_output_tokens = self.initialize_output_tokens_with_length(src_tokens, length_tgt)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def initialize_output_tokens(self, encoder_out, src_tokens):
        if vars(self.args).get("src_upsample_scale", None) is not None or vars(self.args).get("src_upsample_fixed", None) is not None:
            return self.initialize_output_tokens_upsample(encoder_out, src_tokens)
        elif vars(self.args).get("length_multiplier", None) is not None:
            return self.initialize_output_tokens_multiplier(encoder_out, src_tokens)

    def max_positions(self):
        if vars(self.args).get("filter_max_length", None) is not None:
            if ":" not in self.args.filter_max_length:
                a = b = int(self.args.filter_max_length)
            else:
                a, b = self.args.filter_max_length.split(":")
                a, b = int(a), int(b)
            return (a, b)
        else:
            if vars(self.args).get("src_upsample_fixed", None) is not None:
                return (self.encoder.max_positions(), self.decoder.max_positions())
            elif vars(self.args).get("src_upsample_scale", None) is not None:
                return (min(self.encoder.max_positions(), int(self.decoder.max_positions() / self.args.src_upsample_scale)), self.decoder.max_positions())
            else:
                return (min(self.encoder.max_positions(), int(self.decoder.max_positions() / self.args.length_multiplier)), self.decoder.max_positions())

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens

        history = decoder_out.history
        rand_seed = random.randint(0, 19260817)

        # execute the decoder
        output_logits, links, left_tree_mask, right_tree_mask, main_chain = self.extract_features(output_tokens, encoder_out, rand_seed, require_links=True)

        output_logits_normalized = output_logits.log_softmax(dim=-1)
        unreduced_logits, unreduced_tokens_torch = output_logits_normalized.max(dim=-1)
        unreduced_tokens = unreduced_tokens_torch.tolist()
        bsz, prelen, left_span, _ = links.size()

        if self.args.decode_strategy in ["lookahead", "greedy"]:
            if self.args.decode_strategy == "lookahead":
                output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1).tolist()
                links_lookahead_right = links + unreduced_logits.unsqueeze(1).unsqueeze(1) * self.args.decode_beta
                links_lookahead_left = torch.zeros(bsz, prelen, left_span).to(links)
               
                links_lookahead_left = torch.gather(unreduced_logits.unsqueeze(1).repeat(bsz,prelen,1),  dim=2, index=self._max_left_index[:, :prelen, :,].to(links.device)) * self.args.decode_beta
                
                links_lookahead = links_lookahead_right + links_lookahead_left.unsqueeze(-1)
                links_idx = links_lookahead.view(bsz, prelen, left_span*prelen).max(dim=-1)[1].cpu().tolist() # batch * prelen
            elif self.args.decode_strategy == "greedy":
                output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1).tolist()
                links_idx = links.view(bsz, prelen, -1).max(dim=-1)[1].cpu().tolist() # batch * prelen

            unpad_output_tokens = []
            count_last = 0
            for i, length in enumerate(output_length):

                now = 1
                now_node = BinaryTreeNode(now)
                root_node = now_node
                tree_stack = []
                stack = []
                res = []
                path = []
                while (now < length and now != 0) or len(stack) > 0:

                    while now < length and now != 0:
                        stack.append(now)
                        
                        tree_stack.append(now_node)

                        next_idx = links_idx[i][now]
                        links_left_idx = next_idx // prelen
                        links_left_idx = 0 if links_left_idx==0 else now - links_left_idx
                        now_node.leftChild = BinaryTreeNode(links_left_idx)
                        now_node = now_node.leftChild
                        if left_tree_mask[now][links_left_idx]:
                            break
                        
                        now = links_left_idx
                        
                        
                    now = stack.pop()
                    now_node = tree_stack.pop()
                    now_token = unreduced_tokens[i][now]

                    if now_token != self.tgt_dict.pad_index:
                        res.append(now_token)
                    path.append(now)
                    now_node.order = (now_node.order, self.tgt_dict[now_token])

                    
                    
                    next_idx = links_idx[i][now]
                    links_right_idx = next_idx % prelen
                    now = links_right_idx
                    now_node.rightChild = BinaryTreeNode(links_right_idx)
                    now_node = now_node.rightChild
                                    
                unpad_output_tokens.append(res)
            output_seqlen = max([len(res) for res in unpad_output_tokens])
            output_tokens = [res + [self.tgt_dict.pad_index] * (output_seqlen - len(res)) for res in unpad_output_tokens]
            output_tokens = torch.tensor(output_tokens, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)
        elif self.args.decode_strategy in ["viterbi"]:
            output_length = torch.sum(output_tokens.ne(self.tgt_dict.pad_index), dim=-1)

            scores, R_trace, L_trace, M_trace = cuda_pcfg_viterbi(unreduced_logits, links, output_length.cuda(links.get_device()).long())
            # scores, R_trace, L_trace, M_trace = scores.cpu(), R_trace.cpu(), L_trace.cpu(), M_trace.cpu()
            lengths = torch.arange(prelen//4).unsqueeze(0)
            # print(scores.size(), lengths.size())
            length_penalty = (lengths ** self.args.decode_viterbibeta).cuda(scores.get_device())
            scores = scores / length_penalty

            invalid_masks = scores.isnan()
            scores.masked_fill_(invalid_masks, float("-inf"))

            max_score, pred_length = torch.max(scores[:,1:], dim = -1)
            pred_length = pred_length+1
            
            
            R_trace, L_trace, M_trace = R_trace.cpu(), L_trace.cpu(), M_trace.cpu()
            output_tokens = viterbi_decoding(pred_length.long(), output_length.long(), \
            L_trace.long(), R_trace.long(), M_trace.long(), unreduced_tokens_torch.long(), left_tree_mask.long(), self.tgt_dict.pad_index).to(device=pred_length.device)
        if history is not None:
            history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=torch.full(output_tokens.size(), 1.0),
            attn=None,
            history=history,
        )

class GlatLinkDecoderPCFG(NATransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.init_link_feature(args)

    def init_link_feature(self, args):
        links_feature = self.args.links_feature.split(":")
        links_dim = 0
        if "feature" in links_feature:
            links_dim += args.decoder_embed_dim
        if "position" in links_feature:
            self.link_positional = PositionalEmbedding(args.max_target_positions, args.decoder_embed_dim, self.padding_idx, True)
            links_dim += args.decoder_embed_dim
        elif "sinposition" in links_feature:
            self.link_positional = PositionalEmbedding(args.max_target_positions, args.decoder_embed_dim, self.padding_idx, False)
            links_dim += args.decoder_embed_dim
        else:
            self.link_positional = None

        
        self.query_linear = nn.Linear(links_dim, args.decoder_embed_dim)
        self.key_linear_left = nn.Linear(links_dim, args.decoder_embed_dim)
        self.key_linear_right = nn.Linear(links_dim, args.decoder_embed_dim)
        self.gate_linear = nn.Linear(links_dim, args.decoder_attention_heads)

    @staticmethod
    def add_args(parser):
        pass

@register_model_architecture(
    "glat_decomposed_with_link_two_hands_tri_pcfg", "glat_decomposed_with_link_two_hands_tri_pcfg_6e6d512"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)

@register_model_architecture(
    "glat_decomposed_with_link_two_hands_tri_pcfg", "glat_decomposed_with_link_two_hands_tri_pcfg_base"
)
def base_architecture2(args):
    base_architecture(args)

@register_model_architecture("glat_decomposed_with_link_two_hands_tri_pcfg", "glat_decomposed_with_link_two_hands_tri_pcfg_iwslt_de_en")
def nonautoregressive_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)
