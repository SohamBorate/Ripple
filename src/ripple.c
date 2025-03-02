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

int main(int argc, char **argv) {
    int STATUS = 0;

    if (argc != 3) {
        printf("Usage: ripple [WIDTH] [HEIGHT]\n");
        STATUS = 1;
        return STATUS;
    }

    const int WIDTH = atoi(argv[1]);
    const int HEIGHT = atoi(argv[2]);

    const int num_objects = 5;
    BasePart *objects = malloc(num_objects * sizeof(BasePart));

    RGB color;
    color.red = 200;
    color.green = 30;
    color.blue = 30;

    objects[0].DataType = DATA_TYPE_SPHERE;
    objects[0].sphere = sphere_new(vec3_new(0.0,0.0,6.0), 1.0, color);

    color.red = 100;
    color.green = 100;
    color.blue = 100;

    objects[1].DataType = DATA_TYPE_SPHERE;
    objects[1].sphere = sphere_new(vec3_new(-2.0,-0.4,6.0), 0.6, color);

    color.red = 100;
    color.green = 220;
    color.blue = 255;

    objects[2].DataType = DATA_TYPE_SPHERE;
    objects[2].sphere = sphere_new(vec3_new(2.0,-1.0,7.0), 0.8, color);

    color.red = 146;
    color.green = 112;
    color.blue = 45;

    objects[3].DataType = DATA_TYPE_CUBE;
    objects[3].cube = cube_new(vec3_new(2.0,0.0,2.0), vec3_new(1.0, 1.0, 1.0), vec3_new(-1.0, -1.0, -1.0), color);

    color.red = 56;
    color.green = 112;
    color.blue = 45;

    objects[4].DataType = DATA_TYPE_PLANE;
    objects[4].plane = plane_new(vec3_new(0.0,1.0,0.0), vec3_new(0.0, -2.0, 0.0), color);

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
    vec3 sun_pos = vec3_new(5.0, 10.0, -4.0);

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

    render_scene(origin, sun_pos, num_objects, objects, HEIGHT, WIDTH, pixels);

    // Write to BMP file
    STATUS = write_bmp("output.bmp", WIDTH, HEIGHT, pixels);

    // clean up
    free(objects);
    free(pixels);

    // render time
    clock_t end_time = clock();
    double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("\nRender time: %f seconds\n", total_time);

    return STATUS;
}