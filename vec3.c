#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "vec3.h"

void vec3_print(vec3 *v) {
    printf("Vector3:\n\tMemory Address: %p\n\tValue: (%f %f %f)\n\tMagnitude: %f\n", (void *)v, v->x, v->y, v->z, vec3_magnitude(v));
    return;
}

vec3* vec3_new(float x, float y, float z) {
    vec3 *vector = malloc(sizeof(vec3));
    if (vector == NULL) {
        printf("Memory allocation for Vector3 failed\n");
        return NULL;
    }
    vector->x = x;
    vector->y = y;
    vector->z = z;
    return vector;
}

void vec3_free(vec3 *v) {
    free(v);
    return;
}

float vec3_magnitude(vec3 *v) {
    float mag = sqrt((v->x * v->x) + (v->y * v->y) + (v->z * v->z));
    return mag;
}

vec3 *vec3_unit(vec3 *v) {
    float magnitude = vec3_magnitude(v);
    float x = v->x / magnitude;
    float y = v->y / magnitude;
    float z = v->z / magnitude;
    return vec3_new(x, y, z);
}

vec3* vec3_sum(vec3 *v1, vec3 *v2) {
    float x = v1->x + v2->x;
    float y = v1->y + v2->y;
    float z = v1->z + v2->z;
    return vec3_new(x, y, z);
}

vec3* vec3_difference(vec3 *v1, vec3 *v2) {
    float x = v1->x - v2->x;
    float y = v1->y - v2->y;
    float z = v1->z - v2->z;
    return vec3_new(x, y, z);
}

vec3* vec3_scalar_product(vec3 *v, float s){
    float x = v->x * s;
    float y = v->y * s;
    float z = v->z * s;
    return vec3_new(x, y, z);
}

vec3* vec3_vec3_product(vec3 *v1, vec3 *v2){
    float x = v1->x * v2->x;
    float y = v1->y * v2->y;
    float z = v1->z * v2->z;
    return vec3_new(x, y, z);
}

float vec3_dot_vec3(vec3 *v1, vec3 *v2) {
    float r = (v1->x * v2->x) + (v1->y * v2->y) + (v1->z * v2->z);
    return r;
}

float vec3_dot_scalar(vec3 *v, float s) {
    float r = (v->x * s) + (v->y * s) + (v->z * s);
    return r;
}

vec3* vec3_cross_vec3(vec3 *v1, vec3 *v2) {
    float x = v1->y * v2->z - v1->z * v2->y;
    float y = v1->x * v2->z - v1->z * v2->x;
    float z = v1->x * v2->y - v1->y * v2->x;
    return vec3_new(x, y, z);
}