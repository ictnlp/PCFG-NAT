
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

#define MAIN_CHAIN_I2E(x, max_left) ((x)*((max_left)+1)+1)
#define MAIN_CHAIN_E2I(x, max_left) (((x)-1)/((max_left)+1))
#define LOCAL_TREE_I2E(x, max_left) (((x)==0)?0:((x) + ((x)-1) / (max_left) + 1))
#define LOCAL_TREE_E2I(x, max_left) ((x)==0?0:((x) - ((x) - 1) / ((max_left)+1) - 1))

#define BLOCK_BUCKET 16

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}


template<class scalar_t, int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE, class Accessor1, class Accessor2, class Accessor3, class Accessor4>
__global__ void calculate_S_trace_kernel(
    volatile int *bucket_queue, volatile int *accomplish_queue, volatile int *start_queue,
    Accessor1 S_trace,
    Accessor2 C_trace,
    Accessor3 tree,
    Accessor1 S,
    Accessor2 C,
    Accessor1 match_all,
    Accessor2 links,
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
    while(start_queue[batch_id]< (n+2)*n_seg){
        if(main_thread){
            task_id = atomicAdd((int*)start_queue + batch_id, 1);
            // printf("batch_id:%d task_id:%d addr:%d\n", batch_id, task_id, (int*)start_queue + batch_id);
            seg_id = task_id % n_seg;
            start = task_id/n_seg;
            bool done = false;
            while(!done){
                done = true;
                for(int i=0; i<n_seg; i++){
                    done = done && (accomplish_queue[batch_id*n_seg + i] >= start);
                }
            }
        }
        __threadfence();
        __syncthreads();
        int a_id = seg_id * SEQ_BLOCK_SIZE + threadIdx.y;
        int a = MAIN_CHAIN_I2E(a_id, max_left);
        if(start == 0){
            if(seg_id == 0 && main_thread){
                S_trace[batch_id][0][0] = 1;
                // printf("S_trace: %d 1 0 %f\n", batch_id, S_trace[batch_id][1][0]);
            }
        }
        else{
            int _start = start - 1;
            if (a > 0 && a < m && S_trace[batch_id][a_id][_start]!=0){
                int max_c = -1, max_j = 0, max_b = -1;
                scalar_t maxval = -std::numeric_limits<scalar_t>::infinity();
                for(int c_id = a_id + threadIdx.x+1; MAIN_CHAIN_I2E(c_id, max_left) <m ; c_id+=TRANS_BLOCK_SIZE){
                    int c = MAIN_CHAIN_I2E(c_id, max_left);
                    for(int j=0; j<n-_start && j<max_left+1; j++){
                        int b_start = a-max_left;
                        if(b_start<1) b_start = 1;
                        for(int b = b_start; b<a+1; b++){
                            int _b = b==a?0:b;
                            int b_id = LOCAL_TREE_E2I(_b, max_left);
                            scalar_t temp = links[batch_id][a][a-b][c] + S[batch_id][c_id][_start+j+1] + C[batch_id][b_id][_start][j] + match_all[batch_id][a][_start+j];
                            if (isnan(temp)) temp = -std::numeric_limits<scalar_t>::infinity();
                            if(temp > maxval){ maxval = temp; max_c = c_id; max_j = j; max_b = b_id; }
                        }
                    }
                    
                    
                }
                
                __syncwarp(shfl_mask);
                if_constexpr (TRANS_BLOCK_SIZE > 16) {
                    scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 16, TRANS_BLOCK_SIZE);
                    int next_c = __shfl_down_sync(shfl_mask, max_c, 16, TRANS_BLOCK_SIZE); 
                    int next_j = __shfl_down_sync(shfl_mask, max_j, 16, TRANS_BLOCK_SIZE);
                    int next_b = __shfl_down_sync(shfl_mask, max_b, 16, TRANS_BLOCK_SIZE); 
                    if(nextval > maxval){ maxval = nextval; max_c = next_c; max_j = next_j; max_b = next_b; }}
                if_constexpr (TRANS_BLOCK_SIZE > 8) {
                    scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 8, TRANS_BLOCK_SIZE);
                    int next_c = __shfl_down_sync(shfl_mask, max_c, 8, TRANS_BLOCK_SIZE); 
                    int next_j = __shfl_down_sync(shfl_mask, max_j, 8, TRANS_BLOCK_SIZE);
                    int next_b = __shfl_down_sync(shfl_mask, max_b, 8, TRANS_BLOCK_SIZE); 
                    if(nextval > maxval){ maxval = nextval; max_c = next_c; max_j = next_j; max_b = next_b; }}
                if_constexpr (TRANS_BLOCK_SIZE > 4) {
                    scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 4, TRANS_BLOCK_SIZE);
                    int next_c = __shfl_down_sync(shfl_mask, max_c, 4, TRANS_BLOCK_SIZE); 
                    int next_j = __shfl_down_sync(shfl_mask, max_j, 4, TRANS_BLOCK_SIZE);
                    int next_b = __shfl_down_sync(shfl_mask, max_b, 4, TRANS_BLOCK_SIZE); 
                    if(nextval > maxval){ maxval = nextval; max_c = next_c; max_j = next_j; max_b = next_b; }}
                if_constexpr (TRANS_BLOCK_SIZE > 2) {
                    scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 2, TRANS_BLOCK_SIZE);
                    int next_c = __shfl_down_sync(shfl_mask, max_c, 2, TRANS_BLOCK_SIZE); 
                    int next_j = __shfl_down_sync(shfl_mask, max_j, 2, TRANS_BLOCK_SIZE);
                    int next_b = __shfl_down_sync(shfl_mask, max_b, 2, TRANS_BLOCK_SIZE); 
                    if(nextval > maxval){ maxval = nextval; max_c = next_c; max_j = next_j; max_b = next_b; }}
                if_constexpr (TRANS_BLOCK_SIZE > 1) {
                    scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 1, TRANS_BLOCK_SIZE);
                    int next_c = __shfl_down_sync(shfl_mask, max_c, 1, TRANS_BLOCK_SIZE); 
                    int next_j = __shfl_down_sync(shfl_mask, max_j, 1, TRANS_BLOCK_SIZE);
                    int next_b = __shfl_down_sync(shfl_mask, max_b, 1, TRANS_BLOCK_SIZE); 
                    if(nextval > maxval){ maxval = nextval; max_c = next_c; max_j = next_j; max_b = next_b; }}
                maxval = __shfl_sync(shfl_mask, maxval, 0, TRANS_BLOCK_SIZE);
                max_c = __shfl_sync(shfl_mask, max_c, 0, TRANS_BLOCK_SIZE);
                max_j = __shfl_sync(shfl_mask, max_j, 0, TRANS_BLOCK_SIZE);
                max_b = __shfl_sync(shfl_mask, max_b, 0, TRANS_BLOCK_SIZE);

                if(threadIdx.x == 0){
                    // printf("new a:%d start:%d max_b:%d max_j:%d max_c:%d\n", a ,_start, max_b, max_j, max_c);
                    if(max_c!=-1 && max_b!=-1){
                        S_trace[batch_id][max_c][_start+max_j+1] = 1;
                        C_trace[batch_id][max_b][_start][max_j] = 1;
                    }
                    tree[batch_id][a] = _start+max_j;
                    // if(a==1){
                    //     printf("%d %d\n", _start, max_j);
                    // }
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
void invoke_calculate_S_trace(cudaStream_t stream, torch::Tensor &S_trace, torch::Tensor &C_trace, torch::Tensor &tree, const torch::Tensor &S, const torch::Tensor &C, const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, \
    int bsz, int prelen, int tarlen, int max_left)
{
    int main_chain_size = (prelen - 2) / (max_left + 1) + 1;
    int local_tree_size = prelen - main_chain_size;
    int n_seg = (main_chain_size - 1) / SEQ_BLOCK_SIZE + 1;
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
        match_all.scalar_type(), "invoke_calculate_S_trace", [&] {
            tree.fill_(-1);
            calculate_S_trace_kernel<scalar_t, TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
                bucket_queue, accomplish_queue, start_queue,
                S_trace.packed_accessor64<scalar_t, 3>(),
                C_trace.packed_accessor64<scalar_t, 4>(),
                tree.packed_accessor64<int64_t, 2>(),
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


template<class scalar_t, int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE, class Accessor1, class Accessor2, class Accessor3, class Accessor4>
__global__ void calculate_C_kernel_trace(
    Accessor1 C_trace,
    Accessor2 tree,
    Accessor1 C,
    Accessor3 match_all,
    Accessor1 links,
    Accessor4 output_length,
    Accessor4 target_length,
    int bsz, int prelen, int tarlen, int max_left, int n_seg)
{
    bool main_thread = threadIdx.x == 0 && threadIdx.y == 0;


    // int ticket_no = bucket_no * BLOCK_BUCKET + bucket_idx;
    int ticket_no = blockIdx.y;
    int batch_id = ticket_no % bsz;
    int seg_id = ticket_no / bsz;
    int a_id = seg_id * SEQ_BLOCK_SIZE + threadIdx.y + 1;
    int a = LOCAL_TREE_I2E(a_id, max_left);
    int max_left_a = ((a-1) / (max_left+1)) * (max_left+1) + 1;
    int max_right_a = ((a-1) / (max_left+1) +1) * (max_left+1) + 1;
    int m = output_length[batch_id];
    int n = target_length[batch_id];
    // if(main_thread){
    //     printf("batch_id: %d, seg_id: %d, started\n", batch_id, seg_id);
    // }


    for(int gap = max_left+1; gap >= 1; gap--){

        if (a > 0 && a < m){
            for (int i=threadIdx.x; i<n-gap-1; i+=TRANS_BLOCK_SIZE){
                if (C_trace[batch_id][a_id][i][gap]==0) continue;
                int max_j=0, max_b=-1, max_c=-1;
                scalar_t maxval = -std::numeric_limits<scalar_t>::infinity();
                for(int j=0;j<gap;j++){
                    int b_start = a-max_left;
                    if(b_start<1) b_start = 1;
                    for(int b=b_start; b<a+1; b++){
                        int _b = b==a?0:b;
                        int b_id = LOCAL_TREE_E2I(_b, max_left);
                        for(int c=a+1; c<max_right_a && c<m; c++){
                            int c_id = LOCAL_TREE_E2I(c, max_left);
                            scalar_t temp = links[batch_id][a][a-b][c] + C[batch_id][b_id][i][j] + C[batch_id][c_id][i+j+1][gap-j-1] + match_all[batch_id][a][i+j];
                            if (isnan(temp)) temp =   -std::numeric_limits<scalar_t>::infinity();
                            if(temp > maxval){maxval = temp; max_j=j; max_b=b_id, max_c=c_id;}
                        }
                    }
                }
                if(max_b!=-1 && max_c!=-1){
                    C_trace[batch_id][max_b][i][max_j] = 1;
                    C_trace[batch_id][max_c][i+max_j+1][gap-max_j-1] = 1;
                }
                tree[batch_id][a] = i+max_j;
                
                
            }    
        }
        __threadfence();
        __syncthreads();
    }
}


template<int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE>
void invoke_calculate_C_trace(cudaStream_t stream,  torch::Tensor &C_trace, torch::Tensor &tree, const torch::Tensor &C, const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, \
    int bsz, int prelen, int tarlen, int max_left)
{
    if (max_left==0) return;
    int main_chain_size = (prelen - 2) / (max_left + 1) + 1;
    int local_tree_size = prelen - main_chain_size;
    int n_seg = (local_tree_size - 1) / SEQ_BLOCK_SIZE + 1;
    dim3 dimGrid(1, n_seg * bsz);
    dim3 dimBlock(TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE);
    static_assert(TRANS_BLOCK_SIZE <= 32, "TRANS_BLOCK_SIZE should be less than warp size");

    AT_DISPATCH_FLOATING_TYPES(
        match_all.scalar_type(), "invoke_calculate_C_trace", [&] {
            calculate_C_kernel_trace<scalar_t, TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
                C_trace.packed_accessor64<scalar_t, 4>(),
                tree.packed_accessor64<int64_t, 2>(),
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

torch::Tensor pcfg_best_tree(const torch::Tensor &S, const torch::Tensor &C, const torch::Tensor &match_all, const torch::Tensor &links, 
    const torch::Tensor &output_length, const torch::Tensor &target_length,
    int config)
{

    auto bsz = match_all.size(0);
    auto prelen = match_all.size(1);
    auto tarlen = match_all.size(2);
    auto max_left = links.size(2);
    max_left = max_left - 1;
    int main_chain_size = (prelen - 2) / (max_left + 1) + 1;
    int local_tree_size = prelen - main_chain_size;
    torch::Tensor S_trace = at::zeros({bsz, main_chain_size, tarlen+2}, match_all.options());
    torch::Tensor C_trace = at::zeros({bsz, local_tree_size, tarlen+2, max_left+2}, match_all.options());

    torch::Tensor tree = at::zeros({bsz, prelen}, output_length.options());
    cudaStream_t current_stream = 0;
    switch(config){
        case 1: invoke_calculate_S_trace<4, 128>(current_stream, S_trace, C_trace, tree, S, C,  match_all, links, output_length, target_length, bsz, prelen, tarlen, max_left); break;
        default: TORCH_CHECK(config <= 4 && config >= 1, "config should be 1~4");
    }
    switch(config){
        case 1: invoke_calculate_C_trace<4, 128>(current_stream, C_trace, tree, C, match_all, links, output_length, target_length, bsz, prelen, tarlen, max_left); break;
        default: TORCH_CHECK(config <= 4 && config >= 1, "config should be 1~4");
    }

    return tree;
}


