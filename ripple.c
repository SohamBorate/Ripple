#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "render.h"

int main(int argc, char **argv) {
    int STATUS = 0;

    if (argc != 3) {
        printf("Usage: ripple [WIDTH] [HEIGHT]\n");
        STATUS = 1;
        return STATUS;
    }

    const int WIDTH = atoi(argv[1]);
    const int HEIGHT = atoi(argv[2]);

    vec3 *ball_pos = vec3_new(0.0,0.0,10.0);
    if (ball_pos == NULL) {
        STATUS = 3;
        return STATUS;
    }

    RGB color;
    color.red = 112;
    color.green = 1;
    color.blue = 185;
    sphere *ball = sphere_new(ball_pos, 4.0, color);
    if (ball == NULL) {
        STATUS = 4;
        vec3_free(ball_pos);
        return STATUS;
    }

    sphere_print(ball);

    // total time taken
    clock_t start_time = clock();

    vec3 *origin = vec3_new(0.0,0.0,0.0);
    if (origin == NULL) {
        sphere_free(ball);
        STATUS = 5;
        return STATUS;
    }

    vec3 *sun_pos = vec3_new(0.0, 10.0,0.0);
    if (origin == NULL) {
        sphere_free(ball);
        vec3_free(origin);
        STATUS = 6;
        return STATUS;
    }
    printf("Sun position: ");
    vec3_print(sun_pos);

    printf("\nWidth: %i, Height: %i\n", WIDTH, HEIGHT);

    // dynamic memory allocation for 2D array
    RGB (*pixels)[WIDTH] = calloc(HEIGHT, WIDTH * sizeof(RGB));
    // IMPORTANT !!!
    // pixels[row][column]

    if (pixels == NULL) {
        printf("Memory allocation for Pixels failed\n");
        sphere_free(ball);
        vec3_free(origin);
        STATUS = 7;
        return STATUS;
    }

    printf("Light Intensity: %i\n", LIGHT_INTENSITY);
    printf("Field Of View: %i\n", FIELD_OF_VIEW);
    printf("Shading Enabled: %i\n", SHADE);

    render_scene(origin, sun_pos, ball, HEIGHT, WIDTH, pixels);

    // Write to BMP file
    STATUS = write_bmp("output.bmp", WIDTH, HEIGHT, pixels);

    // clean up
    free(pixels);
    vec3_free(origin);
    vec3_free(sun_pos);
    sphere_free(ball);

    // render time
    clock_t end_time = clock();
    double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("\nRender time: %f seconds\n", total_time);

    return STATUS;
}