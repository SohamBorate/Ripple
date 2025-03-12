#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "render.h"

// TODO Raycast module
// TODO Cube

CUDA_DEV void render_scene(vec3 origin, vec3 sun, int num_objects, BasePart *objects, int HEIGHT, int WIDTH, RGB *pixels) {
    const float ASPECT_RATIO = (float) WIDTH / (float) HEIGHT;
    const float H_FOV = ((float) FIELD_OF_VIEW) * (M_PI / 180.0);
    const float V_FOV = H_FOV / ASPECT_RATIO;

    const float SCENE_WIDTH = 2.0 * tan(H_FOV / 2);
    const float SCENE_HEIGHT = 2.0 * tan(V_FOV / 2);

    const float LEAST_WIDTH = SCENE_WIDTH / (float) WIDTH;
    const float LEAST_HEIGHT = SCENE_HEIGHT / (float) HEIGHT;

    printf("\nAspect Ratio: %f\nHorizontal Field Of View in radians: %f\nVertical Field Of View in radians: %f\n", ASPECT_RATIO, H_FOV, V_FOV);
    printf("Scene Width: %f Scene Height: %f\n", SCENE_WIDTH, SCENE_HEIGHT);
    printf("Least Scene Width: %f Least Scene Height: %f\n\n", LEAST_WIDTH, LEAST_HEIGHT);

    for (int row = 0; row < HEIGHT; row++) {
        for (int column = 0; column < WIDTH; column++) {
            RGB pixel = render_pixel(origin, sun, num_objects, objects, row, column, SCENE_HEIGHT, SCENE_WIDTH, LEAST_HEIGHT, LEAST_WIDTH);
            pixels[row * WIDTH + column].blue = pixel.blue;
            pixels[row * WIDTH + column].green = pixel.green;
            pixels[row * WIDTH + column].red = pixel.red;
        }
    }
}

CUDA_DEV RGB render_pixel(vec3 origin, vec3 sun, int num_objects, BasePart *objects, const int ROW, const int COLUMN, const float SCENE_HEIGHT, const float SCENE_WIDTH, const float LEAST_HEIGHT, const float LEAST_WIDTH) {
    RGB pixel;
    pixel.red = 135 * 0.5;
    pixel.green = 206 * 0.5;
    pixel.blue = 235 * 0.5;

    // physical position of the pixel in the scene
    vec3 scene_pixel = vec3_new(SCENE_WIDTH / 2.0 - ( (float)COLUMN + 0.5 ) * LEAST_WIDTH,
                                SCENE_HEIGHT / 2.0 - ( (float)ROW + 0.5 ) * LEAST_HEIGHT,
                                1
                            );

    vec3 direction = vec3_unit(vec3_difference(scene_pixel, origin));

    // printf("Pixel (%i, %i)\n", COLUMN, ROW);

    RaycastResult raycast1 = raycast(origin, direction, num_objects, objects, -1);

    if (raycast1.BasePartIndex == -1) {
        return pixel;
    }

    if (!SHADING_ENABLED) {
        pixel.red = raycast1.color.red;
        pixel.green = raycast1.color.green;
        pixel.blue = raycast1.color.blue;
        return pixel;
    }

    vec3 hit_pos_to_sun = vec3_difference(sun, raycast1.hit_position);

    RaycastResult raycast2 = raycast(raycast1.hit_position, vec3_unit(hit_pos_to_sun), num_objects, objects, raycast1.BasePartIndex);

    // shadows
    if (raycast2.BasePartIndex != -1) {
        if (raycast2.hit_distance <= vec3_magnitude(hit_pos_to_sun)) {
            pixel.red = 0;
            pixel.green = 0;
            pixel.blue = 0;
            return pixel;
        }
    }

    float diffusion = vec3_dot_vec3(raycast1.hit_normal, vec3_unit(hit_pos_to_sun));
    float distance_squared = (float)pow(vec3_magnitude(hit_pos_to_sun), 2);
    float intensity = (float)LIGHT_INTENSITY / (4.0 * M_PI * distance_squared);
    // float intensity = 1;
    // diffusion = fmax(0.0, diffusion);

    pixel.red = validate_rgb((int) ( ((float)(raycast1.color.red)) *  diffusion * intensity * 0.675 ) );
    pixel.green = validate_rgb((int) ( ((float)(raycast1.color.green)) * diffusion * intensity * 0.525 ) );
    pixel.blue = validate_rgb((int) ( ((float)(raycast1.color.blue)) * diffusion * intensity * 0.475 ) );

    // pixel.red = validate_rgb((int) ( ((float)(raycast1.color.red)) *  diffusion * intensity ) );
    // pixel.green = validate_rgb((int) ( ((float)(raycast1.color.green)) * diffusion * intensity ) );
    // pixel.blue = validate_rgb((int) ( ((float)(raycast1.color.blue)) * diffusion * intensity ) );

    // pixel.red = raycast1.color.red;
    // pixel.green = raycast1.color.green;
    // pixel.blue = raycast1.color.blue;

    return pixel;
}

CUDA_DEV uint8_t validate_rgb(int n) {
    if (n < 0) {
        return 0;
    } else if (n > 255) {
        return 255;
    }
    return n;
}