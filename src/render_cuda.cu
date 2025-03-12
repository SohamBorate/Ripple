#include "render.h"

__global__ void render_kernel(vec3 origin, vec3 sun, int num_objects, BasePart *objects, int HEIGHT, int WIDTH, const float SCENE_HEIGHT, const float SCENE_WIDTH, const float LEAST_HEIGHT, const float LEAST_WIDTH, RGB *pixels) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (column >= WIDTH || row >= HEIGHT) return; // Prevent out-of-bounds access

    int index = row * WIDTH + column; // 1D index for pixel array
    pixels[index] = render_pixel(origin, sun, num_objects, objects, row, column, SCENE_HEIGHT, SCENE_WIDTH, LEAST_HEIGHT, LEAST_WIDTH);
}

extern "C" void render_scene_CUDA(vec3 origin_cpu, vec3 sun_cpu, int num_objects, BasePart *objects_cpu, int HEIGHT, int WIDTH, RGB *pixels_cpu) {
    vec3 origin_gpu = origin_cpu;
    vec3 sun_gpu = sun_cpu;

    BasePart *objects_gpu;

    cudaMalloc((void**)&objects_gpu, num_objects * sizeof(BasePart));
    cudaMemcpy(objects_gpu, objects_cpu, num_objects * sizeof(BasePart), cudaMemcpyHostToDevice);

    RGB *pixels_gpu;

    cudaMalloc((void**)&pixels_gpu, HEIGHT * WIDTH * sizeof(RGB));
    cudaMemcpy(pixels_gpu, pixels_cpu, HEIGHT * WIDTH * sizeof(RGB), cudaMemcpyHostToDevice);

    const float ASPECT_RATIO = (float) WIDTH / (float) HEIGHT;
    const float H_FOV = ((float) FIELD_OF_VIEW) * (M_PI / 180.0);
    const float V_FOV = H_FOV / ASPECT_RATIO;

    const float SCENE_WIDTH = 2.0 * tan(H_FOV / 2);
    const float SCENE_HEIGHT = 2.0 * tan(V_FOV / 2);

    const float LEAST_WIDTH = SCENE_WIDTH / (float) WIDTH;
    const float LEAST_HEIGHT = SCENE_HEIGHT / (float) HEIGHT;

    printf("\nReporting from the GPU\n\n");

    // 256 threads per block
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    render_kernel<<<numBlocks, threadsPerBlock>>>(origin_gpu, sun_gpu, num_objects, objects_gpu, HEIGHT, WIDTH, SCENE_HEIGHT, SCENE_WIDTH, LEAST_HEIGHT, LEAST_WIDTH, pixels_gpu);
    
    cudaDeviceSynchronize();

    cudaMemcpy(pixels_cpu, pixels_gpu, HEIGHT * WIDTH * sizeof(RGB), cudaMemcpyDeviceToHost);
    cudaFree(pixels_gpu);
    cudaFree(objects_gpu);
}

// GPU use karne kaa tareeka thoda casual hai
#include "render.c"
