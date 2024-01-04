#include <iostream>
#include <vector>
#include <torch/extension.h>
using namespace std;

std::tuple<torch::Tensor, torch::Tensor> pcfg_loss(const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, int config);
std::tuple<torch::Tensor, torch::Tensor> pcfg_loss_backward(const torch::Tensor &grad_output, const torch::Tensor &S, const torch::Tensor &C, const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, int config);
torch::Tensor pcfg_best_tree(const torch::Tensor &S, const torch::Tensor &C, const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, int config);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> pcfg_viterbi(const torch::Tensor &ob_lprob, const torch::Tensor &links, const torch::Tensor &output_length,int config);
torch::Tensor logsoftmax_gather(torch::Tensor word_ins_out, const torch::Tensor &select_idx, bool require_gradient);

torch::Tensor viterbi_decoding(torch::Tensor pred_length, torch::Tensor output_length, 
                            torch::Tensor L_trace, 
                            torch::Tensor R_trace, 
                            torch::Tensor M_trace, 
                            torch::Tensor unreduced_tokens, 
                            torch::Tensor left_tree_mask, 
                            int pad_index) 
{
    auto batch_size = pred_length.size(0);

    vector <vector <int>> unpad_output_tokens;
    for (int i = 0; i < batch_size; i++)
    {

        int pred_len = pred_length[i].item<int>();
        pair<int, int> now = make_pair(1, pred_len);
        vector <pair<int, int>> stack;
        vector <int> res;
        int max_h = output_length[i].item<int>();
        int last = -1;
        while ((now.first < max_h && now.first != 0) || stack.size() > 0)
        {
            while (now.first < max_h && now.first != 0)
            {
                stack.push_back(now);
                auto links_left_idx = L_trace.index({i, now.first, now.second}).item<int>();
                links_left_idx = links_left_idx == now.first ? 0 : links_left_idx;
                if (left_tree_mask.index({now.first, links_left_idx}).item<int>())
                {
                    break;
                }
                auto now_length = M_trace.index({i, now.first, now.second}).item<int>();
                now = make_pair(links_left_idx, now_length);
            }
            now = stack.back();
            stack.pop_back();
            auto now_token = unreduced_tokens.index({i, now.first}).item<int>();
            if (now_token != pad_index && now_token != last)
            {
                last = now_token;
                res.push_back(now_token);
            }
            auto links_right_idx = R_trace.index({i, now.first, now.second}).item<int>();
            auto now_length = now.second - M_trace.index({i, now.first, now.second}).item<int>() - 1;
            now = make_pair(links_right_idx, now_length);
        }
        unpad_output_tokens.push_back(res);
    }
    int output_seqlen = 0;
    for (int i = 0; i < batch_size; i++)
    {
        output_seqlen = max(output_seqlen, (int)unpad_output_tokens[i].size());
    }
    torch::Tensor output_tokens_tensor = torch::empty({batch_size, output_seqlen}).fill_(pad_index);
    for (int i = 0; i < batch_size; i++)
    {
        output_tokens_tensor.index_put_({i}, torch::tensor(unpad_output_tokens[i]));
    }
    
    return output_tokens_tensor;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pcfg_loss", &pcfg_loss, "PCFG Loss");
  m.def("pcfg_loss_backward", &pcfg_loss_backward, "PCFG Loss Backward");
  m.def("pcfg_best_tree", &pcfg_best_tree, "PCFG Best Tree");
  m.def("pcfg_viterbi", &pcfg_viterbi, "PCFG Viterbi");
  m.def("logsoftmax_gather", &logsoftmax_gather, "logsoftmax + gather");
  m.def("viterbi_decoding", &viterbi_decoding, "Viterbi");
}
