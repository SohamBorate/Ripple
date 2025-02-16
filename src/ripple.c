#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "config.h"
#include "sphere.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FPS 120
#define SPEED (2.0 / 3.0) * M_PI // radians per second
#define LENGTH 10 // seconds

extern void render_scene_CUDA(vec3 origin, vec3 sun, sphere ball, int HEIGHT, int WIDTH, RGB *pixels);

int main(int argc, char **argv) {
    int STATUS = 0;

    if (argc != 3) {
        printf("Usage: ripple [WIDTH] [HEIGHT]\n");
        STATUS = 1;
        return STATUS;
    }

    const int WIDTH = atoi(argv[1]);
    const int HEIGHT = atoi(argv[2]);

    vec3 ball_pos = vec3_new(0.0,0.0,10.0);
    // if (ball_pos == NULL) {
    //     STATUS = 3;
    //     return STATUS;
    // }

    RGB color;
    color.red = 112;
    color.green = 1;
    color.blue = 185;
    sphere ball = sphere_new(ball_pos, 4.0, color);
    // if (ball == NULL) {
    //     STATUS = 4;
    //     vec3_free(ball_pos);
    //     return STATUS;
    // }

    sphere_print(ball);

    vec3 origin = vec3_new(0.0,0.0,0.0);
    // if (origin == NULL) {
    //     sphere_free(ball);
    //     STATUS = 5;
    //     return STATUS;
    // }

    vec3 sun_pos = vec3_new(0.0, 10.0,0.0);
    // if (origin == NULL) {
    //     sphere_free(ball);
    //     vec3_free(origin);
    //     STATUS = 6;
    //     return STATUS;
    // }
    printf("Sun position: ");
    vec3_print(sun_pos);

    printf("\nWidth: %i, Height: %i\n", WIDTH, HEIGHT);

    // dynamic memory allocation for 2D array
    RGB *pixels = malloc(HEIGHT * WIDTH * sizeof(RGB));
    // IMPORTANT !!!
    // pixels[row + column]

    if (pixels == NULL) {
        printf("Memory allocation for Pixels failed\n");
        STATUS = 7;
        return STATUS;
    }

    printf("Light Intensity: %i\n", LIGHT_INTENSITY);
    printf("Field Of View: %i\n", FIELD_OF_VIEW);
    printf("Shading Enabled: %i\n\n", SHADE);

    // total time taken
    clock_t start_time = clock();

    float angle = 0.0;
    int done = 0;

    for (int i = 0; i < FPS * LENGTH; i++) {
        // frame number i
        char name[100];
        sprintf(name, "frames\\output%04d.bmp", i);
        
        sun_pos.x = 17.5 * cos(angle);
        sun_pos.y = 1.0 * sin(angle * 4.0);
        sun_pos.z = 10.0 + 15.0 * sin(angle);

        render_scene_CUDA(origin, sun_pos, ball, HEIGHT, WIDTH, pixels);

        STATUS = write_bmp(name, WIDTH, HEIGHT, pixels);

        angle += (1.0 / FPS) * SPEED;

        if (angle >= 2 * M_PI) {
            angle -= 2 * M_PI;
        }

        done++;
        if (done % FPS == 0) {
            printf("Frames rendered: %i / %i\n", done, FPS * LENGTH);
        }
    }

    // clean up
    free(pixels);

    // render time
    clock_t end_time = clock();
    double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("\nRender time: %f seconds\n", total_time);

    return STATUS;
}