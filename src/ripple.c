#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "base_part.h"
#include "bmp.h"
#include "config.h"
#include "data_types.h"
#include "render.h"
#include "sphere.h"
#include "vec3.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SPEED M_PI / 2.0
#define FPS 120
#define DELTA_TIME 1.0 / (float) FPS
#define LENGTH 10

extern void render_scene_CUDA(vec3 origin, vec3 sun, int num_objects, BasePart *objects, int HEIGHT, int WIDTH, RGB *pixels);

int main(int argc, char **argv) {
    int STATUS = 0;

    if (argc != 3) {
        printf("Usage: ripple [WIDTH] [HEIGHT]\n");
        STATUS = 1;
        return STATUS;
    }

    const int WIDTH = atoi(argv[1]);
    const int HEIGHT = atoi(argv[2]);

    const int num_objects = 3;
    BasePart *objects = malloc(num_objects * sizeof(BasePart));

    RGB color;
    color.red = 200;
    color.green = 30;
    color.blue = 30;

    objects[0].DataType = DATA_TYPE_SPHERE;
    objects[0].sphere = sphere_new(vec3_new(-1.0,-0.5,6.0), 0.8, color);

    // color.red = 100;
    // color.green = 100;
    // color.blue = 100;

    // objects[1].DataType = DATA_TYPE_SPHERE;
    // objects[1].sphere = sphere_new(vec3_new(-2.0,-0.4,6.0), 0.01, color);

    // color.red = 100;
    // color.green = 220;
    // color.blue = 255;

    // objects[2].DataType = DATA_TYPE_SPHERE;
    // objects[2].sphere = sphere_new(vec3_new(2.0,-1.0,7.0), 0.8, color);

    color.red = 146;
    color.green = 112;
    color.blue = 45;

    objects[1].DataType = DATA_TYPE_CUBE;
    // objects[0].cube = cube_new(vec3_new(-4.0,-1.2,10.0), vec3_new(0.5, 0.5, 0.5), vec3_new(-0.5, -0.5, -0.5), color);
    objects[1].cube = cube_new(vec3_new(2.0,-1.0,5.0), vec3_new(0.5, 0.5, 0.5), vec3_new(-0.5, -0.5, -0.5), color);

    color.red = 56;
    color.green = 112;
    color.blue = 45;

    objects[2].DataType = DATA_TYPE_PLANE;
    objects[2].plane = plane_new(vec3_new(0.0,1.0,0.0), vec3_new(0.0, -2.0, 0.0), color);

    for (int i = 0; i < num_objects; i++) {
        if (objects[i].DataType == DATA_TYPE_SPHERE) {
            sphere_print(objects[i].sphere);
        } else if (objects[i].DataType == DATA_TYPE_CUBE) {
            cube_print(objects[i].cube);
        } else if (objects[i].DataType == DATA_TYPE_PLANE) {
            plane_print(objects[i].plane);
        }
    }

    // total time taken
    clock_t start_time = clock();

    vec3 origin = vec3_new(0.0,0.0,0.0);
    vec3 sun_pos = vec3_new(0.0,10.0,0.0);

    printf("Sun position: ");
    vec3_print(sun_pos);

    printf("\nWidth: %i, Height: %i\n", WIDTH, HEIGHT);

    // dynamic memory allocation for 1D array
    RGB *pixels = malloc(HEIGHT * WIDTH * sizeof(RGB));
    // IMPORTANT !!!
    // pixels[row + column]
    // IMPORTANT !!!

    if (pixels == NULL) {
        printf("Memory allocation for Pixels failed\n");
        STATUS = 7;
        return STATUS;
    }

    printf("Light Intensity: %i\n", LIGHT_INTENSITY);
    printf("Field Of View: %i\n", FIELD_OF_VIEW);
    printf("Shading Enabled: %i\n", SHADING_ENABLED);

    float theta = 0;

    // ffmpeg -framerate 120 -i C:\Users\soham\Documents\Programming\Ripple\frames\output_%04d.bmp -c:v libx264 -crf 0 -preset veryslow -pix_fmt yuv444p output.mp4

    for (int i = 0; i < FPS * LENGTH; i++) {

        sun_pos.x = 5.0 * sin(theta);
        sun_pos.z = 5.5 - cos(theta) * 5.5;

        render_scene_CUDA(origin, sun_pos, num_objects, objects, HEIGHT, WIDTH, pixels);

        char file_name[100];
        sprintf(&file_name, "C:\\Users\\soham\\Documents\\Programming\\Ripple\\frames\\output_%04d.bmp", i);

        // Write to BMP file
        STATUS = write_bmp(file_name, WIDTH, HEIGHT, pixels);

        if ((i + 1) % FPS == 0) {
            printf("%i/%i frames rendered\n", i + 1, FPS * LENGTH);
        }

        theta += SPEED * DELTA_TIME;
        if (theta >= 2 * M_PI) {
            theta -= 2 * M_PI;
        }
    }

    // clean up
    free(objects);
    free(pixels);

    // render time
    clock_t end_time = clock();
    double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("\nRender time: %f seconds\n", total_time);

    return STATUS;
}