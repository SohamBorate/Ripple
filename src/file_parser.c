#include "file_parser.h"

void read_scene_file(const FILE *SCENE_FILE, BasePart *OBJECTS, vec3 *SUN_POS) {
    char *s[100];
    int i = 0;
    while (fgets(&s, sizeof(s), SCENE_FILE))
    {
        printf("Line %i: %s", i, s);
        i++;
    }

    return;
}
