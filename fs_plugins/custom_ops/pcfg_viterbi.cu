#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <cassert>
#include <tuple>
#include <type_traits>

#include <c10/macros/Macros.h>
#include <ATen/ATen.h>
#include <cuda.h>

#include <torch/extension.h>
#include <torch/torch.h>
#include "utilities.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(x.type().is_cpu(), #x " must be a CPU tensor")

#define BLOCK_BUCKET 16

template<class scalar_t, int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE, class Accessor1, class Accessor2, class Accessor3, class Accessor4, class Accessor5>
__global__ void calculate_S_kernel(
    volatile int *bucket_queue, volatile int *accomplish_queue,
    Accessor1 S,
    Accessor1 C,
    Accessor2 R,
    Accessor2 L,
    Accessor2 M,
    Accessor3 ob_lprob,
    Accessor4 links,
    Accessor5 output_length,
    int bsz, int prelen,  int max_left, int n_seg)
{
    int bucket_idx = blockIdx.y % BLOCK_BUCKET;
    __shared__ volatile int bucket_no;

    bool main_thread = threadIdx.x == 0 && threadIdx.y == 0;
    if (main_thread){
        // obtain task id
        bucket_no = atomicAdd((int*)bucket_queue + bucket_idx, 1);
    }
    __syncthreads();

    int ticket_no = bucket_no * BLOCK_BUCKET + bucket_idx;
    int batch_id = ticket_no % bsz;

    
    int m = output_length[batch_id];

    int seg_id = ticket_no / bsz;
    int id = seg_id * SEQ_BLOCK_SIZE + threadIdx.y;
    int a = id*(max_left+1)+1;

    // start = 0
    {
        if(seg_id == 0 && main_thread){
            S[batch_id][m-1][1] = ob_lprob[batch_id][m-1];
        }

        __threadfence();
        __syncthreads();
        if(main_thread){
            atomicAdd((int*)accomplish_queue + batch_id, 1);
        }
    }
    for(int length = 2; length < m/4; length++){
        if (main_thread){
            while(accomplish_queue[batch_id] < (length-1)*n_seg); // wait for previous segment to accomplish
        }
        __syncthreads();
        if (a > 0 && a < m){
            scalar_t maxval = -std::numeric_limits<scalar_t>::infinity();
            int max_b=0, max_c=0, max_j=0;
            for(int c_id = id + threadIdx.x; c_id*(max_left+1)+1 <m; c_id+=TRANS_BLOCK_SIZE){
                int c = c_id*(max_left+1)+1;

                for(int j=0; j<max_left+1; j++){
                    int b_start = a-max_left;
                    if(b_start<1) b_start = 1;
                    for(int b = b_start; b<a+1; b++){
                        int _b = b==a?0:b;
                        scalar_t temp = links[batch_id][a][a-b][c] + C[batch_id][_b][j] + S[batch_id][c][length-j-1]  + ob_lprob[batch_id][a];
                        if (isnan(temp)) temp = -std::numeric_limits<scalar_t>::infinity();

                        if(temp > maxval){maxval = temp; max_b=_b; max_c = c; max_j=j;}
                    }
                }
            }
            unsigned shfl_mask = __activemask();
            if_constexpr (TRANS_BLOCK_SIZE > 16) {
                scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 16, TRANS_BLOCK_SIZE);
                int next_c = __shfl_down_sync(shfl_mask, max_c, 16, TRANS_BLOCK_SIZE); 
                int next_b = __shfl_down_sync(shfl_mask, max_b, 16, TRANS_BLOCK_SIZE); 
                int next_j = __shfl_down_sync(shfl_mask, max_j, 16, TRANS_BLOCK_SIZE); 
                if(nextval > maxval){ maxval = nextval; max_c = next_c;  max_b = next_b; max_j = next_j;}}
            if_constexpr (TRANS_BLOCK_SIZE > 8) {
                scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 8, TRANS_BLOCK_SIZE);
                int next_c = __shfl_down_sync(shfl_mask, max_c, 8, TRANS_BLOCK_SIZE); 
                int next_b = __shfl_down_sync(shfl_mask, max_b, 8, TRANS_BLOCK_SIZE);
                int next_j = __shfl_down_sync(shfl_mask, max_j, 8, TRANS_BLOCK_SIZE); 
                if(nextval > maxval){ maxval = nextval; max_c = next_c; max_b = next_b; max_j = next_j;}}
            if_constexpr (TRANS_BLOCK_SIZE > 4) {
                scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 4, TRANS_BLOCK_SIZE);
                int next_c = __shfl_down_sync(shfl_mask, max_c, 4, TRANS_BLOCK_SIZE); 
                int next_b = __shfl_down_sync(shfl_mask, max_b, 4, TRANS_BLOCK_SIZE);
                int next_j = __shfl_down_sync(shfl_mask, max_j, 4, TRANS_BLOCK_SIZE); 
                if(nextval > maxval){ maxval = nextval; max_c = next_c; max_b = next_b; max_j = next_j;}}
            if_constexpr (TRANS_BLOCK_SIZE > 2) {
                scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 2, TRANS_BLOCK_SIZE);
                int next_c = __shfl_down_sync(shfl_mask, max_c, 2, TRANS_BLOCK_SIZE); 
                int next_b = __shfl_down_sync(shfl_mask, max_b, 2, TRANS_BLOCK_SIZE);
                int next_j = __shfl_down_sync(shfl_mask, max_j, 2, TRANS_BLOCK_SIZE); 
                if(nextval > maxval){ maxval = nextval; max_c = next_c;  max_b = next_b; max_j = next_j;}}
            if_constexpr (TRANS_BLOCK_SIZE > 1) {
                scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 1, TRANS_BLOCK_SIZE);
                int next_c = __shfl_down_sync(shfl_mask, max_c, 1, TRANS_BLOCK_SIZE); 
                int next_b = __shfl_down_sync(shfl_mask, max_b, 1, TRANS_BLOCK_SIZE); 
                int next_j = __shfl_down_sync(shfl_mask, max_j, 1, TRANS_BLOCK_SIZE); 
                if(nextval > maxval){ maxval = nextval; max_c = next_c;  max_b = next_b; max_j = next_j;}}
            maxval = __shfl_sync(shfl_mask, maxval, 0, TRANS_BLOCK_SIZE);
            max_c = __shfl_sync(shfl_mask, max_c, 0, TRANS_BLOCK_SIZE);
            max_b = __shfl_sync(shfl_mask, max_b, 0, TRANS_BLOCK_SIZE);
            max_j = __shfl_sync(shfl_mask, max_j, 0, TRANS_BLOCK_SIZE);
            if(threadIdx.x == 0 && !isinf(maxval)){
                
                S[batch_id][a][length] = maxval;
                L[batch_id][a][length] = max_b;
                R[batch_id][a][length] = max_c;
                M[batch_id][a][length] = max_j;
            }
        }
        __threadfence();
        __syncthreads();
        if (main_thread){
            atomicAdd((int*)accomplish_queue + batch_id, 1);
        }
    }

}


template<int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE>
void invoke_calculate_S(cudaStream_t stream, torch::Tensor &S, torch::Tensor &C, torch::Tensor &R, torch::Tensor &L, torch::Tensor &M, const torch::Tensor &ob_lprob, const torch::Tensor &links, const torch::Tensor &output_length, \
    int bsz, int prelen, int max_left)
{
    int n_seg = ((prelen-2) / (max_left+1) + 1 - 1) / SEQ_BLOCK_SIZE + 1;

    dim3 dimGrid(1, n_seg * bsz);
    dim3 dimBlock(TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE);
    // assert(n_seg <= BLOCK_BUCKET);
    int *bucket_queue, *accomplish_queue;
    auto tmp_tensor = at::zeros({BLOCK_BUCKET + bsz}, ob_lprob.options().dtype(at::kInt));
    bucket_queue = tmp_tensor.data_ptr<int>();
    accomplish_queue = bucket_queue + BLOCK_BUCKET;
    static_assert(TRANS_BLOCK_SIZE <= 32, "TRANS_BLOCK_SIZE should be less than warp size");
    AT_DISPATCH_FLOATING_TYPES(
        ob_lprob.scalar_type(), "invoke_calculate_S", [&] {
            S.fill_(-std::numeric_limits<scalar_t>::infinity());
            calculate_S_kernel<scalar_t, TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
                bucket_queue, accomplish_queue,
                S.packed_accessor64<scalar_t, 3>(),
                C.packed_accessor64<scalar_t, 3>(),
                R.packed_accessor64<int64_t, 3>(),
                L.packed_accessor64<int64_t, 3>(),
                M.packed_accessor64<int64_t, 3>(),
                ob_lprob.packed_accessor64<scalar_t, 2>(),
                links.packed_accessor64<scalar_t, 4>(),
                output_length.packed_accessor64<int64_t, 1>(),
                bsz, prelen, max_left, n_seg
            );
        }
    );
}


template<class scalar_t, int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE, class Accessor1, class Accessor2, class Accessor3, class Accessor4, class Accessor5>
__global__ void calculate_C_kernel(
    Accessor1 C,
    Accessor2 R,
    Accessor2 L,
    Accessor2 M,
    Accessor3 ob_lprob,
    Accessor4 links,
    Accessor5 output_length,
    int bsz, int prelen, int max_left, int n_seg)
{

    bool main_thread = threadIdx.x == 0 && threadIdx.y == 0;


    // int ticket_no = bucket_no * BLOCK_BUCKET + bucket_idx;
    int ticket_no = blockIdx.y;
    int batch_id = ticket_no % bsz;
    int seg_id = ticket_no / bsz;
    int a = seg_id * SEQ_BLOCK_SIZE + threadIdx.y + 1;
    int max_left_a = ((a-1) / (max_left+1)) * (max_left+1) + 1;
    int max_right_a = ((a-1) / (max_left+1) +1) * (max_left+1) + 1;
    int m = output_length[batch_id];

    // start = 0
    {
        if(a > 0 && a < m && ((a-max_left_a) % 2 == 1) && threadIdx.x==0){
            C[batch_id][a][1] = ob_lprob[batch_id][a];
        }
        if(seg_id==0 && main_thread){
            C[batch_id][0][0] = 0;
        }
            
        
        __threadfence();
        __syncthreads();
    }
    for(int gap = 2; gap < max_left+1; gap++){

        if (a > 0 && a < m && (a % (max_left+1)!=1)){
            scalar_t maxval = -std::numeric_limits<scalar_t>::infinity();
            int max_b=0, max_c = 0, max_j=0;
            for(int j=0;j<gap;j++){
                int b_start = a-max_left;
                if(b_start<1) b_start = 1;
                for(int b=b_start; b<a+1; b++){
                    int _b = b==a?0:b;
                    for(int c=a+1; c<max_right_a && c<m; c++){
                        scalar_t temp = links[batch_id][a][a-b][c] + C[batch_id][_b][j] + C[batch_id][c][gap-j-1] + ob_lprob[batch_id][a];
                        if (isnan(temp)) temp =   -std::numeric_limits<scalar_t>::infinity();
                        if(temp > maxval){maxval = temp; max_b=_b; max_c=c; max_j=j;}
                    }
                }
            }
            if(!isinf(maxval)){
                C[batch_id][a][gap] = maxval;
                R[batch_id][a][gap] = max_c;
                L[batch_id][a][gap] = max_b;
                M[batch_id][a][gap] = max_j;
            }    
        }
        __threadfence();
        __syncthreads();
    }
}


template<int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE>
void invoke_calculate_C(cudaStream_t stream, torch::Tensor &C, torch::Tensor &R, torch::Tensor &L, torch::Tensor &M, const torch::Tensor &ob_lprob, const torch::Tensor &links, const torch::Tensor &output_length, \
    int bsz, int prelen, int max_left)
{
    
    int n_seg = (prelen - 1) / SEQ_BLOCK_SIZE + 1;
    dim3 dimGrid(1, n_seg * bsz);
    dim3 dimBlock(1, SEQ_BLOCK_SIZE);
    static_assert(TRANS_BLOCK_SIZE <= 32, "TRANS_BLOCK_SIZE should be less than warp size");

    AT_DISPATCH_FLOATING_TYPES(
        ob_lprob.scalar_type(), "invoke_calculate_C", [&] {
            C.fill_(-std::numeric_limits<scalar_t>::infinity());
            calculate_C_kernel<scalar_t, TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
                C.packed_accessor64<scalar_t, 3>(),
                R.packed_accessor64<int64_t, 3>(),
                L.packed_accessor64<int64_t, 3>(),
                M.packed_accessor64<int64_t, 3>(),
                ob_lprob.packed_accessor64<scalar_t, 2>(),
                links.packed_accessor64<scalar_t, 4>(),
                output_length.packed_accessor64<int64_t, 1>(),
                bsz, prelen, max_left, n_seg
            );
        }
    );

}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> pcfg_viterbi(const torch::Tensor &ob_lprob, const torch::Tensor &links, 
    const torch::Tensor &output_length,
    int config)
{
   

    auto bsz = ob_lprob.size(0);
    auto prelen = ob_lprob.size(1);
    auto max_left = links.size(2);
    max_left = max_left - 1;

    
    
    torch::Tensor S = at::zeros({bsz, prelen, prelen/4}, ob_lprob.options());
    torch::Tensor C = at::zeros({bsz, prelen, prelen/4}, ob_lprob.options());
    torch::Tensor R = at::zeros({bsz, prelen, prelen/4}, output_length.options());
    torch::Tensor L = at::zeros({bsz, prelen, prelen/4}, output_length.options());
    torch::Tensor M = at::zeros({bsz, prelen, prelen/4}, output_length.options());
    cudaStream_t current_stream = 0;
    switch(config){
        case 1: invoke_calculate_C<1, 256>(current_stream, C, R, L, M, ob_lprob, links, output_length, bsz, prelen, max_left); break;
        default: TORCH_CHECK(config <= 4 && config >= 1, "config should be 1~4");
    }
    switch(config){
        case 1: invoke_calculate_S<4, 128>(current_stream, S, C, R, L, M, ob_lprob, links, output_length, bsz, prelen, max_left); break;
        default: TORCH_CHECK(config <= 4 && config >= 1, "config should be 1~4");
    }

    return std::make_tuple(S, R, L, M);
}


