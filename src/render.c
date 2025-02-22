#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "render.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void render_scene(vec3 origin, vec3 sun, int num_objects, BasePart *objects, int HEIGHT, int WIDTH, RGB *pixels) {
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

RGB render_pixel(vec3 origin, vec3 sun, int num_objects, BasePart *objects, const int ROW, const int COLUMN, const float SCENE_HEIGHT, const float SCENE_WIDTH, const float LEAST_HEIGHT, const float LEAST_WIDTH) {
    // printf("Row: %i Column: %i\n", ROW, COLUMN);

    RGB pixel;
    pixel.red = 0;
    pixel.green = 0;
    pixel.blue = 0;

    // physical position of the pixel in the scene
    vec3 scene_pixel = vec3_new(SCENE_WIDTH / 2.0 - ( (float)COLUMN + 0.5 ) * LEAST_WIDTH,
                                SCENE_HEIGHT / 2.0 - ( (float)ROW + 0.5 ) * LEAST_HEIGHT,
                                1
                            );

    // vec3_print(scene_pixel);

    vec3 direction = vec3_unit(vec3_difference(scene_pixel, origin));
    int object_index = -1;
    float hit_pos_dist = 10000000000;
    vec3 hit_position;

    // raycasting
    for (int i = 0; i < num_objects; i++) {
        
        // printf("Object DataType: %s\n", objects[i].DataType);
        if (objects[i].DataType == DATA_TYPE_SPHERE) {
            sphere ball = objects[i].sphere;

            // sphere_print(ball);

            vec3 d_vec3 = vec3_difference(ball.position, origin);
            float d_mag = vec3_magnitude(d_vec3);

            // perpendicular distance
            float cos_theta = vec3_dot_vec3(d_vec3, direction) / d_mag;
            const float temp_perp_dist = (float)sqrt(pow(d_mag, 2) * (1 - pow(cos_theta, 2)));

            // printf("Row: %i Column: %i Object: %i A\n", ROW, COLUMN, i);

            // does not hit
            if (temp_perp_dist > ball.radius) {
                continue;
            }

            // printf("Row: %i Column: %i Object: %i B\n", ROW, COLUMN, i);

            const float temp_hit_pos_dist = d_mag * cos_theta - (float)sqrt(pow(ball.radius, 2) - pow(temp_perp_dist, 2));
            vec3 temp_hit_position = vec3_scalar_product(direction, temp_hit_pos_dist);

            if (temp_hit_pos_dist < hit_pos_dist && temp_hit_pos_dist > 0) {
                object_index = i;
                hit_pos_dist = temp_hit_pos_dist;
                hit_position = temp_hit_position;
            }
        }
    }

    if (object_index == -1) {
        return pixel;
    }

    const sphere ball = objects[object_index].sphere;

    // sphere_print(ball);

    if (!SHADING_ENABLED) {
        pixel.red = ball.color.red;
        pixel.green = ball.color.green;
        pixel.blue = ball.color.blue;
        return pixel;
    }

    vec3 normal = vec3_difference(hit_position, ball.position);
    vec3 hit_pos_to_sun = vec3_difference(sun, hit_position);

    int shadow = 0;

    // raycasting while ignoring primary object
    for (int i = 0; i < num_objects; i++) {
        // printf("Object DataType: %s\n", objects[i].DataType);
        if (objects[i].DataType == DATA_TYPE_SPHERE && i != object_index) {
            sphere ball2 = objects[i].sphere;

            // sphere_print(ball);

            vec3 d_vec3 = vec3_difference(ball2.position, hit_position);
            float d_mag = vec3_magnitude(d_vec3);

            // perpendicular distance
            float cos_theta = vec3_dot_vec3(d_vec3, vec3_unit(hit_pos_to_sun)) / d_mag;
            const float temp_perp_dist = (float)sqrt(pow(d_mag, 2) * (1 - pow(cos_theta, 2)));

            // printf("Row: %i Column: %i Object: %i A\n", ROW, COLUMN, i);

            // does not hit
            if (temp_perp_dist > ball2.radius) {
                continue;
            }

            const float temp_hit_pos_dist = d_mag * cos_theta - (float)sqrt(pow(ball.radius, 2) - pow(temp_perp_dist, 2));

            if (temp_hit_pos_dist >= 0) {
                // printf("%i %i\n", i, object_index);
                shadow = 1;
            }

            break;
        }
    }

    if (shadow == 1) {
        // pixel.green = 200;
        return pixel;
    }

    float diffusion = vec3_dot_vec3(normal, hit_pos_to_sun) / (vec3_magnitude(normal) * vec3_magnitude(hit_pos_to_sun));
    float distance_squared = (float)pow(vec3_magnitude(hit_pos_to_sun), 2);
    float intensity = (float)LIGHT_INTENSITY / (4.0 * M_PI * distance_squared);
    // diffusion = fmax(0.0, diffusion);

    pixel.red = validate_rgb((int) ( ((float)(ball.color.red)) *  diffusion * intensity * 0.675 ) );
    pixel.green = validate_rgb((int) ( ((float)(ball.color.green)) * diffusion * intensity * 0.525 ) );
    pixel.blue = validate_rgb((int) ( ((float)(ball.color.blue)) * diffusion * intensity * 0.475 ) );

    return pixel;
}

uint8_t validate_rgb(int n) {
    if (n < 0) {
        return 0;
    } else if (n > 255) {
        return 255;
    }
    return n;
}