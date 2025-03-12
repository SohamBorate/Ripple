#ifndef BASE_PART_H
#define BASE_PART_H

#include "data_types.h"
#include "sphere.h"
#include "cube.h"
#include "plane.h"

typedef struct {
    DataType DataType;
    union {
        sphere sphere;
        cube cube;
        plane plane;
    };
} BasePart;

#endif