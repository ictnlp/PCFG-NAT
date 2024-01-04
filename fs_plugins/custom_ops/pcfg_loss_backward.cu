
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
#define EPSILON 1e-5
#define MAIN_CHAIN_I2E(x, max_left) ((x)*((max_left)+1)+1)
#define MAIN_CHAIN_E2I(x, max_left) (((x)-1)/((max_left)+1))
#define LOCAL_TREE_I2E(x, max_left) (((x)==0)?0:((x) + ((x)-1) / (max_left) + 1))
#define LOCAL_TREE_E2I(x, max_left) ((x)==0?0:((x) - ((x) - 1) / ((max_left)+1) - 1))

template<class scalar_t, int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE, class Accessor1, class Accessor2, class Accessor3, class Accessor4>
__global__ void calculate_S_kernel_grad(
    volatile int *bucket_queue, volatile int *accomplish_queue, volatile int *start_queue,
    Accessor1 grad_output,
    Accessor2 S_grad,
    Accessor2 S,
    Accessor3 C,
    Accessor2 match_all,
    Accessor3 links,
    Accessor4 output_length,
    Accessor4 target_length,
    int bsz, int prelen, int tarlen,  int max_left, int n_seg)
{
    __shared__ volatile int task_id;
    __shared__ volatile int seg_id;
    __shared__ volatile int start;

    bool main_thread = threadIdx.x == 0 && threadIdx.y == 0;

    int ticket_no = blockIdx.y;
    int batch_id = ticket_no % bsz;
    
    int m = output_length[batch_id];
    int n = target_length[batch_id];
    __threadfence();
    __syncthreads();
    unsigned shfl_mask = (1 << TRANS_BLOCK_SIZE) - 1;
    shfl_mask =  shfl_mask << (threadIdx.y % (32 / TRANS_BLOCK_SIZE) * TRANS_BLOCK_SIZE);
    

    while(start_queue[batch_id]< (n+1)*n_seg){
        if(main_thread){
            task_id = atomicAdd((int*)start_queue + batch_id, 1);
            // printf("batch_id:%d task_id:%d addr:%d\n", batch_id, task_id, (int*)start_queue + batch_id);
            seg_id = task_id%n_seg;
            start = task_id/n_seg;
            bool done = false;
            while(!done){
                done = true;
                for(int i=0; i<seg_id+1; i++){
                    done = done && (accomplish_queue[batch_id*n_seg + i] >= start);
                }
            }
        }
        __threadfence();
        __syncthreads();
        int c_id = seg_id * SEQ_BLOCK_SIZE + threadIdx.y;
        int c = MAIN_CHAIN_I2E(c_id, max_left);
        if(start == 0){
            if(seg_id==0 && main_thread){
                S_grad[batch_id][0][0] = grad_output[batch_id];
                // printf("batch_id: %d %f\n", batch_id, S[batch_id][1][0]);
            }
        }
        else{
            int _start = start - 1;
            if(c>0 && c<m){
                for(int j=0; j<n-_start && j< max_left+1; j++){
                    scalar_t sumval = 0;
                    for(int a_id=threadIdx.x; MAIN_CHAIN_I2E(a_id, max_left)<c; a_id+=TRANS_BLOCK_SIZE){
                        int a = MAIN_CHAIN_I2E(a_id, max_left);
                        int b_start = a-max_left;
                        if(b_start<1) b_start=1;
                        for(int b=b_start; b<a+1; b++){
                            int _b = b==a?0:b;
                            int b_id = LOCAL_TREE_E2I(_b, max_left);
                            if (!isinf(S[batch_id][c_id][_start+j+1]) && (!isinf(S[batch_id][a_id][_start]))){
                                sumval += S_grad[batch_id][a_id][_start] * exp(links[batch_id][a][a-b][c] + S[batch_id][c_id][_start+j+1] + C[batch_id][b_id][_start][j] + match_all[batch_id][a][_start+j] - S[batch_id][a_id][_start]);
                                // printf("c:%d a:%d b:%d j:%d %f %f %f %f %f %f\n", c, a, b, j, S_grad[batch_id][a_id][_start], links[batch_id][a][a-b][c], S[batch_id][c_id][_start+j+1], C[batch_id][b_id][_start][j], match_all[batch_id][a][_start+j], S[batch_id][a_id][_start]);
                            }
                        }
                    }
                    
                    __syncwarp(shfl_mask);
                    if_constexpr (TRANS_BLOCK_SIZE > 16) sumval += __shfl_down_sync(shfl_mask, sumval, 16, TRANS_BLOCK_SIZE);
                    if_constexpr (TRANS_BLOCK_SIZE > 8) sumval += __shfl_down_sync(shfl_mask, sumval, 8, TRANS_BLOCK_SIZE);
                    if_constexpr (TRANS_BLOCK_SIZE > 4) sumval += __shfl_down_sync(shfl_mask, sumval, 4, TRANS_BLOCK_SIZE);
                    if_constexpr (TRANS_BLOCK_SIZE > 2) sumval += __shfl_down_sync(shfl_mask, sumval, 2, TRANS_BLOCK_SIZE);
                    if_constexpr (TRANS_BLOCK_SIZE > 1) sumval += __shfl_down_sync(shfl_mask, sumval, 1, TRANS_BLOCK_SIZE);
                    if(threadIdx.x==0){
                        // if(sumval !=0 ) printf("batch_id:%d c:%d _start+j+1:%d\n", batch_id, c, _start+j+1);
                        S_grad[batch_id][c_id][_start+j+1] += sumval;
                    }
                }
            }
        }
        __threadfence();
        __syncthreads();
        if (main_thread){
            atomicAdd((int*)accomplish_queue + batch_id*n_seg + seg_id, 1);
        }
    }

}

template<int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE>
void invoke_calculate_S_grad(cudaStream_t stream, const torch::Tensor &grad_output, torch::Tensor &S_grad, const torch::Tensor &S, const torch::Tensor &C, const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, \
    int bsz, int prelen, int tarlen, int max_left)
{
    int main_chain_size = (prelen-2) / (max_left+1) + 1;
    int n_seg = (main_chain_size - 1) / SEQ_BLOCK_SIZE + 1;
    // n_seg = n_seg * 2;
    dim3 dimGrid(1, 2 * n_seg * bsz);
    dim3 dimBlock(TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE);
    // assert(n_seg <= BLOCK_BUCKET);
    int *bucket_queue, *accomplish_queue, *start_queue;
    auto tmp_tensor = at::zeros({BLOCK_BUCKET + bsz * n_seg}, match_all.options().dtype(at::kInt));
    // auto tmp_tensor = at::zeros({BLOCK_BUCKET + bsz}, match_all.options().dtype(at::kInt));
    bucket_queue = tmp_tensor.data_ptr<int>();
    accomplish_queue = bucket_queue + BLOCK_BUCKET;
    auto tmp_tensor3 = at::zeros({bsz}, match_all.options().dtype(at::kInt));
    start_queue = tmp_tensor3.data_ptr<int>();
    static_assert(TRANS_BLOCK_SIZE <= 32, "TRANS_BLOCK_SIZE should be less than warp size");
    AT_DISPATCH_FLOATING_TYPES(
        match_all.scalar_type(), "invoke_calculate_S_grad", [&] {
            calculate_S_kernel_grad<scalar_t, TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
                bucket_queue, accomplish_queue, start_queue,
                grad_output.packed_accessor64<scalar_t, 1>(),
                S_grad.packed_accessor64<scalar_t, 3>(),
                S.packed_accessor64<scalar_t, 3>(),
                C.packed_accessor64<scalar_t, 4>(),
                match_all.packed_accessor64<scalar_t, 3>(),
                links.packed_accessor64<scalar_t, 4>(),
                output_length.packed_accessor64<int64_t, 1>(),
                target_length.packed_accessor64<int64_t, 1>(),
                bsz, prelen, tarlen, max_left, n_seg
            );
        }
    );
}

template<class scalar_t, int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE, class Accessor1, class Accessor2, class Accessor3>
__global__ void calculate_C_kernel_grad_1(
    Accessor1 S_grad,
    Accessor2 C_grad,
    Accessor1 S,
    Accessor2 C,
    Accessor1 match_all,
    Accessor2 links,
    Accessor3 output_length,
    Accessor3 target_length,
    int bsz, int prelen, int tarlen,  int max_left, int n_seg)
{
    // int ticket_no = bucket_no * BLOCK_BUCKET + bucket_idx;
    int ticket_no = blockIdx.y;
    int batch_id = ticket_no % bsz;
    int seg_id = ticket_no / bsz;
    int b_id = seg_id * SEQ_BLOCK_SIZE + threadIdx.y+1;
    int b = LOCAL_TREE_I2E(b_id, max_left);
    int a = ((b-1) / (max_left+1) +1) * (max_left+1) + 1;
    int a_id = MAIN_CHAIN_E2I(a, max_left);
    int m = output_length[batch_id];
    int n = target_length[batch_id];

    for(int start = n-2; start >= 0; start--){
        if (b > 0 && b < m && a>0 && a<m && S_grad[batch_id][a_id][start]!=0 && !isinf(S[batch_id][a_id][start])){
            for(int j=0; j<n-start && j<max_left+1; j++){
                scalar_t sumval = 0;
                scalar_t temp = 0;
                for(int c_id = a_id + threadIdx.x; MAIN_CHAIN_I2E(c_id, max_left) <m ; c_id+=TRANS_BLOCK_SIZE){
                    int c = MAIN_CHAIN_I2E(c_id, max_left);
                    if (!isinf(C[batch_id][b_id][start][j])){
                        temp = S_grad[batch_id][a_id][start] * exp(links[batch_id][a][a-b][c] + S[batch_id][c_id][start+j+1] + C[batch_id][b_id][start][j] + match_all[batch_id][a][start+j] - S[batch_id][a_id][start]);
                        sumval += temp;
                    }
                }
                unsigned shfl_mask = __activemask();
                shfl_mask = __ballot_sync(shfl_mask, true);
                if_constexpr (TRANS_BLOCK_SIZE > 16) sumval += __shfl_down_sync(shfl_mask, sumval, 16, TRANS_BLOCK_SIZE);
                if_constexpr (TRANS_BLOCK_SIZE > 8) sumval += __shfl_down_sync(shfl_mask, sumval, 8, TRANS_BLOCK_SIZE);
                if_constexpr (TRANS_BLOCK_SIZE > 4) sumval += __shfl_down_sync(shfl_mask, sumval, 4, TRANS_BLOCK_SIZE);
                if_constexpr (TRANS_BLOCK_SIZE > 2) sumval += __shfl_down_sync(shfl_mask, sumval, 2, TRANS_BLOCK_SIZE);
                if_constexpr (TRANS_BLOCK_SIZE > 1) sumval += __shfl_down_sync(shfl_mask, sumval, 1, TRANS_BLOCK_SIZE);
                if (threadIdx.x==0){
                    
                    C_grad[batch_id][b_id][start][j] += sumval;
                }
            }
            
        }
    }
}

template<class scalar_t, int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE, class Accessor1, class Accessor2, class Accessor3>
__global__ void calculate_C_kernel_grad_2(
    Accessor1 C_grad,
    Accessor1 C,
    Accessor2 match_all,
    Accessor1 links,
    Accessor3 output_length,
    Accessor3 target_length,
    int bsz, int prelen, int tarlen,  int max_left, int n_seg)
{

    // int ticket_no = bucket_no * BLOCK_BUCKET + bucket_idx;
    int ticket_no = blockIdx.y;
    int batch_id = ticket_no % bsz;
    int seg_id = ticket_no / bsz;
    int selected_id = seg_id * SEQ_BLOCK_SIZE + threadIdx.y;
    int selected_h = LOCAL_TREE_I2E(selected_id, max_left);
    int max_right_a = ((selected_h-1) / (max_left+1) +1) * (max_left+1) + 1;
    int max_left_a = ((selected_h-1) / (max_left+1)) * (max_left+1) + 1;
    
    int m = output_length[batch_id];
    int n = target_length[batch_id];



    for(int gap=max_left+1;gap>=2;gap--){
        if (selected_h > 0 && selected_h < m){
            int b = selected_h;
            for (int a=b+1; a<max_right_a && a<m; a++){
                int a_id = LOCAL_TREE_E2I(a, max_left);
                for (int i=threadIdx.x; i<n-gap; i+=TRANS_BLOCK_SIZE){
                    if (C_grad[batch_id][a_id][i][gap]==0) continue;
                    for(int j=0;j<gap;j++){
                        int _b = (b==a)?0:b;
                        int b_id = LOCAL_TREE_E2I(_b, max_left);
                        for(int c=a+1; c<max_right_a && c<m; c++){
                            int c_id = LOCAL_TREE_E2I(c, max_left);
                            if(!isinf(C[batch_id][b_id][i][j]))
                                C_grad[batch_id][b_id][i][j] += C_grad[batch_id][a_id][i][gap]*exp(links[batch_id][a][a-b][c]+C[batch_id][b_id][i][j]+C[batch_id][c_id][i+j+1][gap-j-1]+match_all[batch_id][a][i+j]-C[batch_id][a_id][i][gap]);
                        }
                    }
                    
                }
            }

            int c = selected_h;
            int c_id = selected_id;
            for (int a=max_left_a+1; a<c; a++){
                int a_id = LOCAL_TREE_E2I(a, max_left);
                for (int i=threadIdx.x; i<n-gap; i+=TRANS_BLOCK_SIZE){
                    if (C_grad[batch_id][a_id][i][gap]==0) continue;
                    for(int j=0;j<gap;j++){
                        for(int b=max_left_a+1; b<a+1; b++){
                            int _b = b==a?0:b;
                            int b_id = LOCAL_TREE_E2I(_b, max_left);
                            if(!isinf(C[batch_id][c_id][i+j+1][gap-j-1]))
                                C_grad[batch_id][c_id][i+j+1][gap-j-1] += C_grad[batch_id][a_id][i][gap]*exp(links[batch_id][a][a-b][c]+C[batch_id][b_id][i][j]+C[batch_id][c_id][i+j+1][gap-j-1]+match_all[batch_id][a][i+j]-C[batch_id][a_id][i][gap]);
                        }
                    }
                }
            }
        }
        
        __threadfence();
        __syncthreads();
    }
}


template<int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE>
void invoke_calculate_C_grad(cudaStream_t stream, torch::Tensor &S_grad, torch::Tensor &C_grad, const torch::Tensor &S, const torch::Tensor &C, const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, \
    int bsz, int prelen, int tarlen, int max_left)
{
    if(max_left == 0) return;
    int main_chain_size = (prelen - 2) / (max_left + 1) + 1;
    int local_tree_size = prelen - main_chain_size;
    int n_seg = (local_tree_size - 1) / SEQ_BLOCK_SIZE + 1;
    dim3 dimGrid(1, n_seg * bsz);
    dim3 dimBlock(TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE);
    // assert(n_seg <= BLOCK_BUCKET);
    static_assert(TRANS_BLOCK_SIZE <= 32, "TRANS_BLOCK_SIZE should be less than warp size");
    AT_DISPATCH_FLOATING_TYPES(
        match_all.scalar_type(), "invoke_calculate_C_grad_1", [&] {
            calculate_C_kernel_grad_1<scalar_t, TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
                S_grad.packed_accessor64<scalar_t, 3>(),
                C_grad.packed_accessor64<scalar_t, 4>(),
                S.packed_accessor64<scalar_t, 3>(),
                C.packed_accessor64<scalar_t, 4>(),
                match_all.packed_accessor64<scalar_t, 3>(),
                links.packed_accessor64<scalar_t, 4>(),
                output_length.packed_accessor64<int64_t, 1>(),
                target_length.packed_accessor64<int64_t, 1>(),
                bsz, prelen, tarlen, max_left, n_seg
            );
        }
    );
    int n_seg2 = (prelen - 1) / SEQ_BLOCK_SIZE + 1;
    dim3 dimGrid2(1, n_seg2 * bsz);
    AT_DISPATCH_FLOATING_TYPES(
        match_all.scalar_type(), "invoke_calculate_C_grad_2", [&] {
            calculate_C_kernel_grad_2<scalar_t, TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE><<<dimGrid2, dimBlock, 0, stream>>>(
                C_grad.packed_accessor64<scalar_t, 4>(),
                C.packed_accessor64<scalar_t, 4>(),
                match_all.packed_accessor64<scalar_t, 3>(),
                links.packed_accessor64<scalar_t, 4>(),
                output_length.packed_accessor64<int64_t, 1>(),
                target_length.packed_accessor64<int64_t, 1>(),
                bsz, prelen, tarlen, max_left, n_seg2
            );
        }
    );

}

template<class scalar_t, int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE, class Accessor1, class Accessor2, class Accessor3>
__global__ void calculate_match_all_kernel_grad_1(
    Accessor1 match_all_grad,
    Accessor1 S_grad,
    Accessor2 C_grad,
    Accessor1 S,
    Accessor2 C,
    Accessor1 match_all,
    Accessor2 links,
    Accessor3 output_length,
    Accessor3 target_length,
    int bsz, int prelen, int tarlen,  int max_left, int n_seg)
{
    bool main_thread = threadIdx.x == 0 && threadIdx.y == 0;
    int ticket_no = blockIdx.y;
    int batch_id = ticket_no % bsz;
    int m = output_length[batch_id];
    int n = target_length[batch_id];

    int seg_id = ticket_no / bsz;
    int a_id = seg_id * SEQ_BLOCK_SIZE + threadIdx.y;
    int a = MAIN_CHAIN_I2E(a_id, max_left);

    for(int start=0;start<n;start++){

        if(a>0 && a<m && S_grad[batch_id][a_id][start]!=0 && !isinf(S[batch_id][a_id][start])){
            for(int j=0; j<n-start && j< max_left+1; j++){
                scalar_t sumval = 0;
                for(int c_id = a_id + threadIdx.x; MAIN_CHAIN_I2E(c_id, max_left) <m ; c_id+=TRANS_BLOCK_SIZE){
                    int c = MAIN_CHAIN_I2E(c_id, max_left);
                    int b_start = a-max_left;
                    if(b_start<1) b_start = 1;
                    for(int b=b_start; b<a+1; b++){
                        int _b = b==a?0:b;
                        int b_id = LOCAL_TREE_E2I(_b, max_left);
                        if (!isinf(match_all[batch_id][a][start+j]) && (!isinf(S[batch_id][a_id][start]))){
                            sumval += S_grad[batch_id][a_id][start] * exp(links[batch_id][a][a-b][c] + S[batch_id][c_id][start+j+1] + C[batch_id][b_id][start][j] + match_all[batch_id][a][start+j] - S[batch_id][a_id][start]);      
                        }
                    }
                    
                }
                unsigned shfl_mask = __activemask();
                shfl_mask = __ballot_sync(shfl_mask, true);
                if_constexpr (TRANS_BLOCK_SIZE > 16) sumval += __shfl_down_sync(shfl_mask, sumval, 16, TRANS_BLOCK_SIZE);
                if_constexpr (TRANS_BLOCK_SIZE > 8) sumval += __shfl_down_sync(shfl_mask, sumval, 8, TRANS_BLOCK_SIZE);
                if_constexpr (TRANS_BLOCK_SIZE > 4) sumval += __shfl_down_sync(shfl_mask, sumval, 4, TRANS_BLOCK_SIZE);
                if_constexpr (TRANS_BLOCK_SIZE > 2) sumval += __shfl_down_sync(shfl_mask, sumval, 2, TRANS_BLOCK_SIZE);
                if_constexpr (TRANS_BLOCK_SIZE > 1) sumval += __shfl_down_sync(shfl_mask, sumval, 1, TRANS_BLOCK_SIZE);
                if(threadIdx.x==0){
                    match_all_grad[batch_id][a][start+j] += sumval;
                }
            }
        }
    }
    if(main_thread && seg_id==0){
        int last_id = MAIN_CHAIN_E2I(m-1, max_left);
        match_all_grad[batch_id][m-1][n-1] += S_grad[batch_id][last_id][n-1];
    }
}

template<class scalar_t, int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE, class Accessor1, class Accessor2, class Accessor3>
__global__ void calculate_match_all_kernel_grad_2(
    Accessor1 match_all_grad,
    Accessor2 C_grad,
    Accessor2 C,
    Accessor1 match_all,
    Accessor2 links,
    Accessor3 output_length,
    Accessor3 target_length,
    int bsz, int prelen, int tarlen,  int max_left, int n_seg)
{
    int ticket_no = blockIdx.y;
    int batch_id = ticket_no % bsz;
    int m = output_length[batch_id];
    int n = target_length[batch_id];

    int seg_id = ticket_no / bsz;
    int a_id = seg_id * SEQ_BLOCK_SIZE + threadIdx.y;
    int a = LOCAL_TREE_I2E(a_id, max_left);
    int max_left_a = ((a-1) / (max_left+1)) * (max_left+1) + 1;
    int max_right_a = ((a-1) / (max_left+1) +1) * (max_left+1) + 1;
    
    for(int gap=max_left+1;gap>2;gap--){
        if(a>0 && a<m){
            for (int i=threadIdx.x; i<n-gap-1; i+=TRANS_BLOCK_SIZE){
                if(C_grad[batch_id][a_id][i][gap]==0 || isinf(C[batch_id][a_id][i][gap])) continue;
                for(int j=0;j<gap;j++){
                    int b_start = a-max_left;
                    if(b_start<1) b_start = 1;
                    for(int b=b_start; b<a+1; b++){
                        int _b = b==a?0:b;
                        int b_id = LOCAL_TREE_E2I(_b, max_left);
                        for(int c=a+1; c<max_right_a && c<m; c++){
                            int c_id = LOCAL_TREE_E2I(c, max_left);
                            if(!isinf(match_all[batch_id][a][i+j])){
                                match_all_grad[batch_id][a][i+j] += C_grad[batch_id][a_id][i][gap]  * exp(links[batch_id][a][a-b][c] + C[batch_id][b_id][i][j] + C[batch_id][c_id][i+j+1][gap-j-1] + match_all[batch_id][a][i+j]-C[batch_id][a_id][i][gap]);
                            }
                        }
                    }
        
                }
            }
        }
    }
    for(int i=threadIdx.x; i<n; i+=TRANS_BLOCK_SIZE){
        if(a>0 && a<m && ((a-max_left_a) % 2 == 1)){
            match_all_grad[batch_id][a][i] += C_grad[batch_id][a_id][i][1];
        }
    }
    // if(threadIdx.x==0 && a==m-1){
    //     match_all_grad[batch_id][a][n-1] += C_grad[batch_id][a][n-1][0];
    // }
    
}


template<int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE>
void invoke_calculate_match_all_grad(cudaStream_t stream, torch::Tensor &match_all_grad, torch::Tensor &S_grad, torch::Tensor &C_grad, const torch::Tensor &S, const torch::Tensor &C, const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, \
    int bsz, int prelen, int tarlen, int max_left)
{
    int main_chain_size = (prelen - 2) / (max_left + 1) + 1;
    int local_tree_size = prelen - main_chain_size;

    int n_seg = (main_chain_size - 1) / SEQ_BLOCK_SIZE + 1;

    dim3 dimGrid(1, n_seg * bsz);
    dim3 dimBlock(TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE);
    // assert(n_seg <= BLOCK_BUCKET);
    static_assert(TRANS_BLOCK_SIZE <= 32, "TRANS_BLOCK_SIZE should be less than warp size");
    AT_DISPATCH_FLOATING_TYPES(
        match_all.scalar_type(), "calculate_match_all_kernel_grad_1", [&] {
            calculate_match_all_kernel_grad_1<scalar_t, TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
                match_all_grad.packed_accessor64<scalar_t, 3>(),
                S_grad.packed_accessor64<scalar_t, 3>(),
                C_grad.packed_accessor64<scalar_t, 4>(),
                S.packed_accessor64<scalar_t, 3>(),
                C.packed_accessor64<scalar_t, 4>(),
                match_all.packed_accessor64<scalar_t, 3>(),
                links.packed_accessor64<scalar_t, 4>(),
                output_length.packed_accessor64<int64_t, 1>(),
                target_length.packed_accessor64<int64_t, 1>(),
                bsz, prelen, tarlen, max_left, n_seg
            );
        }
    );
    int n_seg_2 = (local_tree_size - 1) / SEQ_BLOCK_SIZE + 1;
    dim3 dimGrid2(1, n_seg_2 * bsz);
    AT_DISPATCH_FLOATING_TYPES(
        match_all.scalar_type(), "calculate_match_all_kernel_grad_2", [&] {
            calculate_match_all_kernel_grad_2<scalar_t, TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE><<<dimGrid2, dimBlock, 0, stream>>>(
                match_all_grad.packed_accessor64<scalar_t, 3>(),
                C_grad.packed_accessor64<scalar_t, 4>(),
                C.packed_accessor64<scalar_t, 4>(),
                match_all.packed_accessor64<scalar_t, 3>(),
                links.packed_accessor64<scalar_t, 4>(),
                output_length.packed_accessor64<int64_t, 1>(),
                target_length.packed_accessor64<int64_t, 1>(),
                bsz, prelen, tarlen, max_left, n_seg_2
            );
        }
    );

}


template<class scalar_t, int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE, class Accessor1, class Accessor2, class Accessor3>
__global__ void calculate_links_kernel_grad_1(
    Accessor1 links_grad,
    Accessor2 S_grad,
    Accessor1 C_grad,
    Accessor2 S,
    Accessor1 C,
    Accessor2 match_all,
    Accessor1 links,
    Accessor3 output_length,
    Accessor3 target_length,
    int bsz, int prelen, int tarlen,  int max_left, int n_seg)
{
    int ticket_no = blockIdx.y;
    int batch_id = ticket_no % bsz;
    int m = output_length[batch_id];
    int n = target_length[batch_id];

    int seg_id = ticket_no / bsz;
    int a_id = seg_id * SEQ_BLOCK_SIZE + threadIdx.y;
    int a = MAIN_CHAIN_I2E(a_id, max_left);

    for(int start=0;start<n;start++){
        if(a>0 && a<m){
            if( S_grad[batch_id][a_id][start]==0) continue;
            for(int c_id = a_id + threadIdx.x; MAIN_CHAIN_I2E(c_id, max_left) <m ; c_id+=TRANS_BLOCK_SIZE){
                int c = MAIN_CHAIN_I2E(c_id, max_left);
                int b_start = a-max_left;
                if(b_start<1) b_start = 1;
                for(int b=b_start; b<a+1; b++){
                    int _b = b==a?0:b;
                    int b_id = LOCAL_TREE_E2I(_b, max_left);
                    scalar_t sumval = 0;
                    for(int j=0; j<n-start && j< max_left+1; j++){
                        if((!isinf(links[batch_id][a][a-b][c])) && (!isinf(S[batch_id][a_id][start]))){
                            sumval += S_grad[batch_id][a_id][start] * exp(links[batch_id][a][a-b][c] + S[batch_id][c_id][start+j+1] + C[batch_id][b_id][start][j] + match_all[batch_id][a][start+j] - S[batch_id][a_id][start]);
                        }
                    }
                    links_grad[batch_id][a][a-b][c] += sumval;
                }
            }
        }
    }
}

template<class scalar_t, int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE, class Accessor1, class Accessor2, class Accessor3>
__global__ void calculate_links_kernel_grad_2(
    Accessor2 links_grad,
    Accessor2 C_grad,
    Accessor2 C,
    Accessor1 match_all,
    Accessor2 links,
    Accessor3 output_length,
    Accessor3 target_length,
    int bsz, int prelen, int tarlen,  int max_left, int n_seg)
{
    // int ticket_no = bucket_no * BLOCK_BUCKET + bucket_idx;
    int ticket_no = blockIdx.y;
    int batch_id = ticket_no % bsz;
    int seg_id = ticket_no / bsz;
    int a_id = seg_id * SEQ_BLOCK_SIZE + threadIdx.y+1;
    int a = LOCAL_TREE_I2E(a_id, max_left);
    int max_right = (a / (max_left+1) +1) * (max_left+1) + 1;
    int m = output_length[batch_id];
    int n = target_length[batch_id];

    for(int gap=max_left+1;gap>2;gap--){
        if (a > 0 && a < m && (a % (max_left+1)!=1)){
            
            for(int c=a+1; c<max_right && c<m; c++){
                int c_id = LOCAL_TREE_E2I(c, max_left);
                int b_start = a-max_left;
                if(b_start<1) b_start = 1;
                for(int b=b_start; b<a+1; b++){
                    int _b = b==a?0:b;
                    int b_id = LOCAL_TREE_E2I(_b, max_left);
                    scalar_t sumval = 0;
                    for(int i=threadIdx.x; i<n-gap; i+=TRANS_BLOCK_SIZE){
                        if(C_grad[batch_id][a_id][i][gap]<EPSILON) continue;
                        // if(a==7){
                        //     printf("%d, %f, %f\n",i, C_grad[batch_id][a_id][i][gap], C[batch_id][a_id][i][gap]);
                        // }
                        for(int j=0; j<gap; j++){
                            if((!isinf(links[batch_id][a][a-b][c])) && (!isinf(C[batch_id][a_id][i][gap]))){
                                sumval += C_grad[batch_id][a_id][i][gap]  * exp(links[batch_id][a][a-b][c] + C[batch_id][b_id][i][j] + C[batch_id][c_id][i+j+1][gap-j-1] + match_all[batch_id][a][i+j]-C[batch_id][a_id][i][gap]);
                                // if(a==7 && c==8 && _b==1){
                                //     printf("%d, %f\n",i, C_grad[batch_id][a_id][i][gap]);
                                // }
                                // if(a==7){
                                //     printf("a:%d b:%d c:%d\n",a_id,b_id,c_id);
                                // }
                                
                            }
                        
                        }
                        
                    }
                    unsigned shfl_mask = __activemask();
                    shfl_mask = __ballot_sync(shfl_mask, true);
                    if_constexpr (TRANS_BLOCK_SIZE > 16) sumval += __shfl_down_sync(shfl_mask, sumval, 16, TRANS_BLOCK_SIZE);
                    if_constexpr (TRANS_BLOCK_SIZE > 8) sumval += __shfl_down_sync(shfl_mask, sumval, 8, TRANS_BLOCK_SIZE);
                    if_constexpr (TRANS_BLOCK_SIZE > 4) sumval += __shfl_down_sync(shfl_mask, sumval, 4, TRANS_BLOCK_SIZE);
                    if_constexpr (TRANS_BLOCK_SIZE > 2) sumval += __shfl_down_sync(shfl_mask, sumval, 2, TRANS_BLOCK_SIZE);
                    if_constexpr (TRANS_BLOCK_SIZE > 1) sumval += __shfl_down_sync(shfl_mask, sumval, 1, TRANS_BLOCK_SIZE);
                    if(threadIdx.x==0){
                        links_grad[batch_id][a][a-b][c] += sumval;
                    }
                }

            }
        }

    }
}


template<int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE>
void invoke_calculate_links_grad(cudaStream_t stream, torch::Tensor &links_grad, torch::Tensor &S_grad, torch::Tensor &C_grad, const torch::Tensor &S, const torch::Tensor &C, const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, \
    int bsz, int prelen, int tarlen, int max_left)
{
    int main_chain_size = (prelen - 2) / (max_left + 1) + 1;
    int local_tree_size = prelen - main_chain_size;
    int n_seg = (main_chain_size - 1) / SEQ_BLOCK_SIZE + 1;
    dim3 dimGrid(1, n_seg * bsz);
    dim3 dimBlock(TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE);
    // assert(n_seg <= BLOCK_BUCKET);
    static_assert(TRANS_BLOCK_SIZE <= 32, "TRANS_BLOCK_SIZE should be less than warp size");
    AT_DISPATCH_FLOATING_TYPES(
        match_all.scalar_type(), "calculate_links_kernel_grad_1", [&] {
            calculate_links_kernel_grad_1<scalar_t, TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
                links_grad.packed_accessor64<scalar_t, 4>(),
                S_grad.packed_accessor64<scalar_t, 3>(),
                C_grad.packed_accessor64<scalar_t, 4>(),
                S.packed_accessor64<scalar_t, 3>(),
                C.packed_accessor64<scalar_t, 4>(),
                match_all.packed_accessor64<scalar_t, 3>(),
                links.packed_accessor64<scalar_t, 4>(),
                output_length.packed_accessor64<int64_t, 1>(),
                target_length.packed_accessor64<int64_t, 1>(),
                bsz, prelen, tarlen, max_left, n_seg
            );
        }
    );

    int n_seg_2 = (local_tree_size - 1) / SEQ_BLOCK_SIZE + 1;
    dim3 dimGrid2(1, n_seg_2 * bsz);
    // assert(n_seg <= BLOCK_BUCKET);
    static_assert(TRANS_BLOCK_SIZE <= 32, "TRANS_BLOCK_SIZE should be less than warp size");
    AT_DISPATCH_FLOATING_TYPES(
        match_all.scalar_type(), "invoke_calculate_links_grad_2", [&] {
            calculate_links_kernel_grad_2<scalar_t, TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
                links_grad.packed_accessor64<scalar_t, 4>(),
                C_grad.packed_accessor64<scalar_t, 4>(),
                C.packed_accessor64<scalar_t, 4>(),
                match_all.packed_accessor64<scalar_t, 3>(),
                links.packed_accessor64<scalar_t, 4>(),
                output_length.packed_accessor64<int64_t, 1>(),
                target_length.packed_accessor64<int64_t, 1>(),
                bsz, prelen, tarlen, max_left, n_seg_2
            );
        }
    );
}

std::tuple<torch::Tensor, torch::Tensor> pcfg_loss_backward(const torch::Tensor &grad_output,const torch::Tensor &S, const torch::Tensor &C,
    const torch::Tensor &match_all, const torch::Tensor &links, 
    const torch::Tensor &output_length, const torch::Tensor &target_length,
    int config)
{
    // CHECK_CUDA(match_all);  // bsz * tarlen * prelen
    // CHECK_CUDA(links);   // bsz * prelen * translen
    // CHECK_CUDA(output_length); // bsz
    // CHECK_CUDA(target_length); // bsz
    // TORCH_CHECK(match_all.dim() == 3, "match_all dim != 3");
    // TORCH_CHECK(links.dim() == 4, "links dim != 3");
    // TORCH_CHECK(output_length.dim() == 1, "output_length dim != 3");
    // TORCH_CHECK(target_length.dim() == 1, "target_length dim != 3");

    auto bsz = match_all.size(0);
    auto prelen = match_all.size(1);
    auto tarlen = match_all.size(2);
    auto max_left = links.size(2);
    max_left = max_left - 1;

    int main_chain_size = (prelen - 2) / (max_left + 1) + 1;
    int local_tree_size = prelen - main_chain_size;
    
    // TORCH_CHECK(links.size(0) == bsz && output_length.size(0) == bsz && target_length.size(0) == bsz, "batch size not match");
    // TORCH_CHECK(links.size(1) == prelen, "prelen not match");
    // TORCH_CHECK(output_length.scalar_type() == at::kLong && target_length.scalar_type() == at::kLong, "length should be long");

    
    // printf("alpha0\n");

    // calculate alpha
    // printf("%d %d %d\n", bsz, tarlen, prelen);
    torch::Tensor S_grad = at::zeros({bsz, main_chain_size, tarlen+2}, match_all.options());
    torch::Tensor C_grad = at::zeros({bsz, local_tree_size, tarlen+2, max_left+2}, match_all.options());

    torch::Tensor match_all_grad = at::zeros({bsz, prelen, tarlen}, match_all.options());
    torch::Tensor links_grad = at::zeros({bsz, prelen, max_left+1, prelen}, match_all.options());

    cudaStream_t current_stream = 0;
    // printf("invoke_calculate_S_grad\n");
    switch(config){
        case 1: invoke_calculate_S_grad<4, 128>(current_stream, grad_output, S_grad, S, C, match_all, links, output_length, target_length, bsz, prelen, tarlen, max_left); break;
        default: TORCH_CHECK(config <= 4 && config >= 1, "config should be 1~4");
    }
    // cudaDeviceSynchronize();
    // printf("invoke_calculate_C_grad\n");
    switch(config){
        case 1: invoke_calculate_C_grad<4, 128>(current_stream, S_grad, C_grad, S, C, match_all, links, output_length, target_length, bsz, prelen, tarlen, max_left); break;
        default: TORCH_CHECK(config <= 4 && config >= 1, "config should be 1~4");
    }
    // cudaDeviceSynchronize();
    // printf("invoke_calculate_match_all_grad\n");
    switch(config){
        case 1: invoke_calculate_match_all_grad<4, 128>(current_stream, match_all_grad, S_grad, C_grad, S, C, match_all, links, output_length, target_length, bsz, prelen, tarlen, max_left); break;
        default: TORCH_CHECK(config <= 4 && config >= 1, "config should be 1~4");
    }
    // cudaDeviceSynchronize();
    // printf("invoke_calculate_links_grad\n");
    switch(config){
        case 1: invoke_calculate_links_grad<4, 128>(current_stream, links_grad, S_grad, C_grad, S, C, match_all, links, output_length, target_length, bsz, prelen, tarlen, max_left); break;
        default: TORCH_CHECK(config <= 4 && config >= 1, "config should be 1~4");
    }


    // printf("alpha4\n");
    return std::make_tuple(match_all_grad, links_grad);
    // return std::make_tuple(S_grad, C_grad);
}



