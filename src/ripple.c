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
    color.red = 112;
    color.green = 1;
    color.blue = 185;

    objects[0].DataType = DATA_TYPE_SPHERE;
    objects[0].sphere = sphere_new(vec3_new(0.0,0.0,20.0), 4.0, color);

    color.red = 1;
    color.green = 112;
    color.blue = 185;

    objects[1].DataType = DATA_TYPE_SPHERE;
    objects[1].sphere = sphere_new(vec3_new(3.0,0.0,12.0), 1.0, color);

    color.red = 146;
    color.green = 112;
    color.blue = 185;

    objects[2].DataType = DATA_TYPE_SPHERE;
    objects[2].sphere = sphere_new(vec3_new(-15.0,-2.0,20.0), 10.0, color);

    for (int i = 0; i < num_objects; i++) {
        if (objects[i].DataType == DATA_TYPE_SPHERE) {
            sphere_print(objects[i].sphere);
        }
    }

    // total time taken
    clock_t start_time = clock();

    vec3 origin = vec3_new(0.0,0.0,0.0);
    vec3 sun_pos = vec3_new(10.0, 0.0, 0.0);

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