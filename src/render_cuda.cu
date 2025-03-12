#include "render.h"

__global__ void render_kernel(vec3 origin, vec3 sun, sphere ball, int HEIGHT, int WIDTH, RGB *pixels) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (column >= WIDTH || row >= HEIGHT) return; // Prevent out-of-bounds access

    int index = row * WIDTH + column; // 1D index for pixel array
    pixels[index] = render_pixel(origin, sun, ball, row, column, HEIGHT, WIDTH);
}

extern "C" void render_scene_CUDA(vec3 origin_cpu, vec3 sun_cpu, sphere ball_cpu, int HEIGHT, int WIDTH, RGB *pixels_cpu) {
    vec3 origin_gpu = origin_cpu;
    vec3 sun_gpu = sun_cpu;
    sphere ball_gpu = ball_cpu;

    RGB *pixels_gpu;

    cudaMalloc((void**)&pixels_gpu, HEIGHT * WIDTH * sizeof(RGB));
    cudaMemcpy(pixels_gpu, pixels_cpu, HEIGHT * WIDTH * sizeof(RGB), cudaMemcpyHostToDevice);

    printf("\nReporting from the GPU\n\n");

    sphere_print(ball_gpu);
    printf("Sun position: ");
    vec3_print(sun_gpu);
    printf("\nWidth: %i, Height: %i\n", WIDTH, HEIGHT);
    printf("Light Intensity: %i\n", LIGHT_INTENSITY);
    printf("Field Of View: %i\n", FIELD_OF_VIEW);
    printf("Shading Enabled: %i\n", SHADE);

    // 256 threads per block
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    render_kernel<<<numBlocks, threadsPerBlock>>>(origin_gpu, sun_gpu, ball_gpu, HEIGHT, WIDTH, pixels_gpu);
    
    cudaDeviceSynchronize();

    cudaMemcpy(pixels_cpu, pixels_gpu, HEIGHT * WIDTH * sizeof(RGB), cudaMemcpyDeviceToHost);
    cudaFree(pixels_gpu);
}

// GPU use karne kaa tareeka thoda casual hai
#include "render.c"
