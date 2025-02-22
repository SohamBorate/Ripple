#ifndef BASE_PART_H
#define BASE_PART_H

#include "data_types.h"
#include "sphere.h"

typedef struct {
    DataType DataType;
    union {
        sphere sphere;
    };
} BasePart;

#endif