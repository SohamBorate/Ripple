#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "bmp.h"
#include "sphere.h"

#define LIGHT_INTENSITY 1

void render(vec3 *origin, sphere *ball, int height, int width, RGB pixels[height][width]);

float get_point_from_sphere(vec3 *origin, vec3 *direction, sphere *ball);

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: ripple [width] [height]\n");
        return 0;
    }

    const int width = atoi(argv[1]);
    const int height = atoi(argv[2]);

    if (width != height) {
        printf("Only squares are allowed\n");
        return 0;
    }

    vec3 *vector = vec3_new(0.0,0.0,10.0);
    if (vector == NULL) {return 1;}

    sphere *ball = sphere_new(vector, 4.0);
    if (ball == NULL) {return 2;}

    sphere_print(ball);

    // total time taken
    clock_t start_time = clock();

    vec3 *origin = vec3_new(0.0,0.0,0.0);
    if (origin == NULL) {
        sphere_free(ball);
        return 5;
    }

    printf("\nWidth: %i, Height: %i\n", width, height);

    // dynamic memory allocation for 2D array
    RGB (*pixels)[width] = calloc(height, width * sizeof(RGB));

    if (pixels == NULL) {
        sphere_free(ball);
        vec3_free(origin);
        return 3;
    }

    render(origin, ball, height, width, pixels);

    // Write to BMP file
    int result = write_bmp("output.bmp", width, height, pixels);

    // clean up
    free(pixels);
    vec3_free(origin);
    sphere_free(ball);

    // render time
    clock_t end_time = clock();
    double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("\nRender time: %f seconds\n", total_time);

    return result;
}

void render(vec3 *origin, sphere *ball, int height, int width, RGB pixels[height][width]) {
    const float start_x = 0.5 - (float)((1.0 / width) / 2.0);
    const float start_y = 0.5 - (float)((1.0 / height) / 2.0);
    vec3 *render_pos = vec3_new(origin->x + start_x, origin->y + start_y, origin->z + 0.7);
    vec3 *sun = vec3_new(0.0,-10.0,0.0);
    printf("Sun position: ");
    vec3_print(sun);

    // render
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            vec3 *direction = vec3_difference(render_pos, origin);

            float point = get_point_from_sphere(origin, direction, ball);

            if (point >= 0) {
                // sphere
                // RGB 118, 185, 0
                RGB color;
                color.blue = 0;
                color.green = 185;
                color.red = 118;

                vec3 *hit_pos = vec3_scalar_product(direction, point);
                vec3 *nhit_pos = vec3_difference(hit_pos, ball->position);
                vec3 *normal = vec3_scalar_product(nhit_pos, (float)(1.0 / vec3_magnitude(nhit_pos)));
                vec3 *light = vec3_difference(sun, hit_pos);
                float diffusion = (vec3_dot_vec3(normal,light) / (vec3_magnitude(normal) * vec3_magnitude(light)));

                pixels[i][j].blue = validate_rgb((int)(color.blue + (color.blue * diffusion * LIGHT_INTENSITY)));
                pixels[i][j].green = validate_rgb((int)(color.green + (color.green * diffusion * LIGHT_INTENSITY)));
                pixels[i][j].red = validate_rgb((int)(color.red + (color.red * diffusion * LIGHT_INTENSITY)));

                vec3_free(hit_pos);
                vec3_free(nhit_pos);
                vec3_free(normal);
                vec3_free(light);
            } else {
                // background
                // RGB 0, 0, 0
                pixels[i][j].blue = validate_rgb(0);
                pixels[i][j].green = validate_rgb(0);
                pixels[i][j].red = validate_rgb(0);
            }

            vec3_free(direction);
            // move 1 pixel to the right
            render_pos->x = render_pos->x - (float)(1.0 / (width));
        }
        // move 1 pixel to the bottom start
        render_pos->x = origin->x + start_x;
        render_pos->y = render_pos->y - (float)(1.0 / (height));
    }

    vec3_free(render_pos);
    vec3_free(sun);
}

float get_point_from_sphere(vec3 *origin, vec3 *direction, sphere *ball) {
    vec3 *ball_to_origin = vec3_difference(origin, ball->position);
    if (vec3_magnitude(ball_to_origin) <= ball->radius) {return -1;}

    // quadratic equation
    float a = pow(vec3_magnitude(direction), 2);
    float b = 2 * vec3_dot_vec3(direction, ball_to_origin);
    float c = pow(vec3_magnitude(ball_to_origin), 2) - pow(ball->radius, 2);
    float D = pow(b,2) - (4*a*c);

    vec3_free(ball_to_origin);
    if (!(D >= 0)) {return D;}

    // quadratic formula
    float x1 = (((-1 * b) + sqrt(D)) / (2 * a));
    float x2 = (((-1 * b) - sqrt(D)) / (2 * a));

    return fmin(x1, x2);
}