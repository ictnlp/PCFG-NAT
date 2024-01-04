
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

#define MAIN_CHAIN_I2E(x, max_left) (x)*((max_left)+1)+1
#define MAIN_CHAIN_E2I(x, max_left) ((x)-1)/((max_left)+1)
#define LOCAL_TREE_I2E(x, max_left) (((x)==0)?0:((x) + ((x)-1) / (max_left) + 1))
#define LOCAL_TREE_E2I(x, max_left) (((x)==0)?0:((x) - ((x) - 1) / ((max_left)+1) - 1))

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
    cudaError err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "last cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    err = cudaPeekAtLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "peek last cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.

#endif

    return;
}


template<class scalar_t, int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE, class Accessor1, class Accessor2, class Accessor3, class Accessor4>
__global__ void calculate_S_kernel(
    volatile int *bucket_queue, volatile int *accomplish_queue, volatile int *start_queue,
    Accessor1 S,
    Accessor2 C,
    Accessor3 match_all,
    Accessor2 links,
    Accessor4 output_length,
    Accessor4 target_length,
    int bsz, int prelen, int tarlen,  int max_left, int n_seg)
{
    int bucket_idx = blockIdx.y % BLOCK_BUCKET;
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
    int src_line = threadIdx.y % (32 / TRANS_BLOCK_SIZE)* TRANS_BLOCK_SIZE;
    src_line = 0;
    int a_id = 0;
    int a = 0;
    bool done = false;
    scalar_t maxval = -std::numeric_limits<scalar_t>::infinity();
    scalar_t temp = -std::numeric_limits<scalar_t>::infinity();
    scalar_t sumval = 0;
    while(start_queue[batch_id] < n*n_seg){
        
        
        if(main_thread){
            task_id = atomicAdd((int*)start_queue + batch_id, 1);
            seg_id = n_seg -1 - (task_id % n_seg);
            start = n-1 - (task_id / n_seg);
            

            done = false;
            while(!done){
                done = true;
                for(int i=seg_id; i<n_seg; i++){
                    done = done && (accomplish_queue[batch_id*n_seg + i] >= n-1-start);
                }
            }
        }
        __threadfence();
        __syncthreads();
        a_id = seg_id * SEQ_BLOCK_SIZE + threadIdx.y;
        a = MAIN_CHAIN_I2E(a_id, max_left);
        if (start == n-1){
            if(main_thread && seg_id==0){
                int last_id = MAIN_CHAIN_E2I(m-1, max_left);
                S[batch_id][last_id][n-1] = match_all[batch_id][m-1][n-1];
            }
        }
        else{ 
            if (a > 0 && a < m){
                maxval = -std::numeric_limits<scalar_t>::infinity();
                temp = -std::numeric_limits<scalar_t>::infinity();
                for(int c_id = a_id + threadIdx.x; c_id*(max_left+1)+1 <m; c_id+=TRANS_BLOCK_SIZE){
                    int c = MAIN_CHAIN_I2E(c_id, max_left);
                    for(int j=0; j<n-start && j<max_left+1; j++){
                        int b_start = a-max_left;
                        if(b_start<1) b_start = 1;
                        for(int b = b_start; b<a+1; b++){
                            int _b = b==a?0:b;
                            int b_id = LOCAL_TREE_E2I(_b, max_left);
                            temp = links[batch_id][a][a-b][c] + S[batch_id][c_id][start+j+1] + C[batch_id][b_id][start][j] + match_all[batch_id][a][start+j];
            
                            if (isnan(temp)) temp = -std::numeric_limits<scalar_t>::infinity();
                            if(temp > maxval) maxval = temp;

                        }
                    }
                    
                    
                }
                // if(a==241 && start == 30){printf("ended x:%d %f\n", threadIdx.x, maxval);}
                

                __syncwarp(shfl_mask);
                if_constexpr (TRANS_BLOCK_SIZE > 16) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 16, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
                if_constexpr (TRANS_BLOCK_SIZE > 8) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 8, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
                if_constexpr (TRANS_BLOCK_SIZE > 4) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 4, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
                if_constexpr (TRANS_BLOCK_SIZE > 2) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 2, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
                if_constexpr (TRANS_BLOCK_SIZE > 1) {scalar_t nextval = __shfl_down_sync(shfl_mask, maxval, 1, TRANS_BLOCK_SIZE); if(nextval > maxval) maxval = nextval;}
                maxval = __shfl_sync(shfl_mask, maxval, src_line, TRANS_BLOCK_SIZE);

                // shfl_mask = __ballot_sync(shfl_mask, !isinf(maxval));
                float res;
                if (isinf(maxval)){
                    res = maxval;
                }
                else{
                    sumval = 0;
                    for(int c_id = a_id + threadIdx.x; c_id*(max_left+1)+1 <m ; c_id+=TRANS_BLOCK_SIZE){
                        int c = MAIN_CHAIN_I2E(c_id, max_left);
                        for(int j=0; j<n-start && j<max_left+1; j++){
                            int b_start = a-max_left;
                            if(b_start<1) b_start = 1;
                            for(int b = b_start; b<a+1; b++){
                                int _b = b==a?0:b;
                                int b_id = LOCAL_TREE_E2I(_b, max_left);
                                temp = links[batch_id][a][a-b][c] + S[batch_id][c_id][start+j+1] + C[batch_id][b_id][start][j] + match_all[batch_id][a][start+j] - maxval;
                                if (isnan(temp)) temp = -std::numeric_limits<scalar_t>::infinity();
                                sumval += exp(temp);
                            }
                        }
                        
                    }
                    __syncwarp(shfl_mask);
                    if_constexpr (TRANS_BLOCK_SIZE > 16) sumval += __shfl_down_sync(shfl_mask, sumval, 16, TRANS_BLOCK_SIZE);
                    if_constexpr (TRANS_BLOCK_SIZE > 8) sumval += __shfl_down_sync(shfl_mask, sumval, 8, TRANS_BLOCK_SIZE);
                    if_constexpr (TRANS_BLOCK_SIZE > 4) sumval += __shfl_down_sync(shfl_mask, sumval, 4, TRANS_BLOCK_SIZE);
                    if_constexpr (TRANS_BLOCK_SIZE > 2) sumval += __shfl_down_sync(shfl_mask, sumval, 2, TRANS_BLOCK_SIZE);
                    if_constexpr (TRANS_BLOCK_SIZE > 1) sumval += __shfl_down_sync(shfl_mask, sumval, 1, TRANS_BLOCK_SIZE);
                    res = log(sumval) + maxval;
                }
                if(threadIdx.x == 0){
                    S[batch_id][a_id][start] = res;
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
void invoke_calculate_S(cudaStream_t stream,torch::Tensor &S, torch::Tensor &C, const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, \
    int bsz, int prelen, int tarlen, int max_left)
{
    int main_chain_size = (prelen-2) / (max_left+1) + 1;
    int n_seg = (main_chain_size - 1) / SEQ_BLOCK_SIZE + 1;
    n_seg = 2*n_seg;
    dim3 dimGrid(1, n_seg * bsz);
    dim3 dimBlock(TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE);
    int *bucket_queue, *accomplish_queue, *start_queue;
    auto tmp_tensor = at::zeros({BLOCK_BUCKET}, match_all.options().dtype(at::kInt));
    bucket_queue = tmp_tensor.data_ptr<int>();
    auto tmp_tensor2 = at::zeros({bsz * n_seg}, match_all.options().dtype(at::kInt));
    accomplish_queue = tmp_tensor2.data_ptr<int>();
    auto tmp_tensor3 = at::zeros({bsz}, match_all.options().dtype(at::kInt));
    start_queue = tmp_tensor3.data_ptr<int>();
    static_assert(TRANS_BLOCK_SIZE <= 32, "TRANS_BLOCK_SIZE should be less than warp size");
    AT_DISPATCH_FLOATING_TYPES(
        match_all.scalar_type(), "invoke_calculate_S", [&] {
            S.fill_(-std::numeric_limits<scalar_t>::infinity());
            calculate_S_kernel<scalar_t, TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
                bucket_queue, accomplish_queue, start_queue,
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
    // CudaCheckError();
}

template<class scalar_t, int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE, class Accessor1, class Accessor2, class Accessor3>
__global__ void calculate_C_kernel(
    Accessor1 C,
    Accessor2 match_all,
    Accessor1 links,
    Accessor3 output_length,
    Accessor3 target_length,
    int bsz, int prelen, int tarlen, int max_left, int n_seg)
{

    bool main_thread = threadIdx.x == 0 && threadIdx.y == 0;


    // int ticket_no = bucket_no * BLOCK_BUCKET + bucket_idx;
    int ticket_no = blockIdx.y;
    int batch_id = ticket_no % bsz;
    int seg_id = ticket_no / bsz;
    int a_id = seg_id * SEQ_BLOCK_SIZE + threadIdx.y;
    int a = LOCAL_TREE_I2E(a_id, max_left);
    int max_left_a = ((a-1) / (max_left+1)) * (max_left+1) + 1;
    int max_right_a = ((a-1) / (max_left+1) +1) * (max_left+1) + 1;
    int m = output_length[batch_id];
    int n = target_length[batch_id];


    // start = 0
    {
        if(main_thread && seg_id == 0){
            C[batch_id][0][n][0] = 0;
        }
        for(int i=threadIdx.x; i<n; i+=TRANS_BLOCK_SIZE){
            if(a > 0 && a < m && ((a-max_left_a) % 2 == 1)){
                C[batch_id][a_id][i][1] = match_all[batch_id][a][i];
            }
            C[batch_id][0][i][0] = 0;
        }
        __threadfence();
        __syncthreads();
    }
    for(int gap = 2; gap < max_left+2; gap++){
        if (a > 0 && a < m){
            for (int i=threadIdx.x; i<n-gap-1; i+=TRANS_BLOCK_SIZE){
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
                            if(temp > maxval) maxval = temp;
                        }
                    }
                }
                float res;
                if (isinf(maxval)){
                    res = maxval;
                }
                else{
                    scalar_t sumval = 0;
                    for(int j=0;j<gap;j++){
                        int b_start = a-max_left;
                        if(b_start<1) b_start = 1;
                        for(int b=b_start; b<a+1; b++){
                            int _b = b==a?0:b;
                            int b_id = LOCAL_TREE_E2I(_b, max_left);
                            for(int c=a+1; c<max_right_a && c<m; c++){
                                int c_id = LOCAL_TREE_E2I(c, max_left);
                                scalar_t temp = links[batch_id][a][a-b][c] + C[batch_id][b_id][i][j] + C[batch_id][c_id][i+j+1][gap-j-1] + match_all[batch_id][a][i+j] - maxval;
                                if (isnan(temp)) temp =   -std::numeric_limits<scalar_t>::infinity();
                                sumval += exp(temp);
                            }
                        }
                    }
                    res = log(sumval) + maxval;
                    
                }
                C[batch_id][a_id][i][gap] = res;
                
            }    
        }
        __threadfence();
        __syncthreads();
    }
}


template<int TRANS_BLOCK_SIZE, int SEQ_BLOCK_SIZE>
void invoke_calculate_C(cudaStream_t stream, torch::Tensor &C, const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, \
    int bsz, int prelen, int tarlen, int max_left)
{
    int main_chain_size = (prelen - 2) / (max_left + 1) + 1;
    int local_tree_size = prelen - main_chain_size;
    int n_seg = (local_tree_size - 1) / SEQ_BLOCK_SIZE + 1;
    dim3 dimGrid(1, n_seg * bsz);
    dim3 dimBlock(TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE);
    static_assert(TRANS_BLOCK_SIZE <= 32, "TRANS_BLOCK_SIZE should be less than warp size");

    AT_DISPATCH_FLOATING_TYPES(
        match_all.scalar_type(), "invoke_calculate_C", [&] {
            C.fill_(-std::numeric_limits<scalar_t>::infinity());
            calculate_C_kernel<scalar_t, TRANS_BLOCK_SIZE, SEQ_BLOCK_SIZE><<<dimGrid, dimBlock, 0, stream>>>(
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

std::tuple<torch::Tensor, torch::Tensor>  pcfg_loss(const torch::Tensor &match_all, const torch::Tensor &links, 
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
    

    torch::Tensor S = at::zeros({bsz, main_chain_size, tarlen+2}, match_all.options());
    torch::Tensor C = at::zeros({bsz, local_tree_size, tarlen+2, max_left+2}, match_all.options());
    cudaStream_t current_stream = 0;

    invoke_calculate_C<4, 128>(current_stream, C, match_all, links, output_length, target_length, bsz, prelen, tarlen, max_left);

    invoke_calculate_S<4, 128>(current_stream, S, C, match_all, links, output_length, target_length, bsz, prelen, tarlen, max_left); 
    return std::make_tuple(S, C);
}


