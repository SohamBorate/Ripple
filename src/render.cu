#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "config.h"
#include "sphere.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define THREADS_PER_BLOCK 256

__device__ uint8_t validate_rgb(int n);

__device__ RGB render_pixel(vec3 origin, vec3 sun, sphere ball, const int ROW, const int COLUMN, const int HEIGHT, const int WIDTH) {
    RGB pixel;
    pixel.red = 0;
    pixel.green = 0;
    pixel.blue = 0;

    vec3 ball_to_sun = vec3_difference(sun, ball.position);

    if (vec3_magnitude(ball_to_sun) < ball.radius) {
        return pixel;
    }

    // physical position of the pixel in the scene
    vec3 scene_pixel = vec3_new(0.1 * ( ( ((float) WIDTH) / 2.0 ) - 0.5 - ( 1.0 * ((float) COLUMN) ) ),
                                0.1 * ( ( ((float) HEIGHT) / 2.0 ) - 0.5 - ( 1.0 * ((float) ROW) ) ),
                                121 - FIELD_OF_VIEW
                            );

    vec3 temp_direction = vec3_difference(scene_pixel, origin);
    vec3 direction = vec3_unit(temp_direction);

    // distance between ball and origin
    vec3 d_vec3 = vec3_difference(ball.position, origin);
    float d_mag = vec3_magnitude(d_vec3);

    // perpendicular distance
    float cos_theta = vec3_dot_vec3(d_vec3, direction) / d_mag;
    const float perp_dist = sqrt(pow(d_mag, 2) * (1 - pow(cos_theta, 2)));

    if (perp_dist > ball.radius) {
        return pixel;
    }    

    if (!SHADE) {
        pixel.red = 115;
        pixel.green = 0;
        pixel.blue = 185;
        return pixel;
    }

    const float hit_pos_dist = d_mag * cos_theta - sqrt(pow(ball.radius, 2) - pow(perp_dist, 2));
    vec3 hit_position = vec3_scalar_product(direction, hit_pos_dist);

    vec3 normal = vec3_difference(hit_position, ball.position);
    vec3 hit_pos_to_sun = vec3_difference(sun, hit_position);

    float diffusion = vec3_dot_vec3(normal, hit_pos_to_sun) / (vec3_magnitude(normal) * vec3_magnitude(hit_pos_to_sun));
    float distance_squared = pow(vec3_magnitude(hit_pos_to_sun), 2);
    float intensity = LIGHT_INTENSITY / (4.0 * M_PI * distance_squared);
    // diffusion = fmax(0.0, fmin(1.0, diffusion));

    pixel.red = validate_rgb((int) ( ((float)(ball.color.red)) *  diffusion * intensity * 0.675 ) );
    pixel.green = validate_rgb((int) ( ((float)(ball.color.green)) * diffusion * intensity * 0.525 ) );
    pixel.blue = validate_rgb((int) ( ((float)(ball.color.blue)) * diffusion * intensity * 0.475 ) );

    return pixel;
}

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

// extern "C" void render_scene_CUDA(vec3 *origin, vec3 *sun, sphere *ball, int HEIGHT, int WIDTH, RGB *pixels) {
//     for (int row = 0; row < HEIGHT; row++) {
//         for (int column = 0; column < WIDTH; column++) {
//             RGB pixel = render_pixel(origin, sun, ball, row, column, HEIGHT, WIDTH);
//             pixels[row][column].blue = pixel.blue;
//             pixels[row][column].green = pixel.green;
//             pixels[row][column].red = pixel.red;
//         }
//     }
// }

__device__ uint8_t validate_rgb(int n) {
    if (n < 0) {
        return 0;
    } else if (n > 255) {
        return 255;
    }
    return n;
}