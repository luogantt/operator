#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_warp_3d()
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int bdx = blockDim.x;
    int bdy = blockDim.y;
    int bdz = blockDim.z;

    // 3D thread 坐标压平成线性 thread id
    int linear_tid = tz * (bdx * bdy) + ty * bdx + tx;

    // warp 信息
    int warp_id_in_block = linear_tid / 32;
    int lane_id = linear_tid % 32;

    printf("Block(%d,%d,%d) Thread(%d,%d,%d) linear_tid=%d warp_id=%d lane_id=%d\n",
           bx, by, bz,
           tx, ty, tz,
           linear_tid, warp_id_in_block, lane_id);
}

int main()
{
    dim3 grid_size(2, 3, 4);
    dim3 block_size(4, 8, 4);

    hello_warp_3d<<<grid_size, block_size>>>();

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
