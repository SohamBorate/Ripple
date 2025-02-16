#include "config.h"
#include "sphere.h"

void render_scene(vec3 *origin, vec3 *sun, sphere *ball, int HEIGHT, int WIDTH, RGB pixels[HEIGHT][WIDTH]);

RGB render_pixel(vec3 *origin, vec3 *sun, sphere *ball, const int ROW, const int COLUMN, const int HEIGHT, const int WIDTH);

float get_point_from_sphere(vec3 *origin, vec3 *direction, sphere *ball);