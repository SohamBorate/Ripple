#ifndef FILE_PARSER_H
#define FILE_PARSER_H

#include <stdio.h>
#include <stdlib.h>

#include "base_part.h"
#include "bmp.h"
#include "config.h"
#include "data_types.h"
#include "render.h"
#include "sphere.h"
#include "string_utils.h"
#include "vec3.h"

void read_scene_file(FILE *SCENE_FILE, BasePart *OBJECTS, vec3 *SUN_POS);

#endif