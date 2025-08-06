#include "file_parser.h"

#define READ_MODES 2
#define READ_SUB_MODES 10

struct mode_config {
    char mode[100];
    char sub_modes[READ_SUB_MODES][100];
};

const struct mode_config MODES[READ_MODES];

// MODES[0].mode = "light_source\n";

// this is gonna take so so long

// MODES[1].mode = "lighbase_partt_source\n";


int match_mode(char input[100]) {
    int mode_index = -1;
    for (int i = 0; i < READ_MODES; i++) {
        if (strcmp(input, MODES[i].mode) == 0) {
            mode_index = i;
            break;
        }
    }
    return mode_index;
}

int match_sub_mode(char input[100]) {
    int mode_index = -1;
    return mode_index;
}

void read_scene_file(FILE *SCENE_FILE, BasePart *OBJECTS, vec3 *SUN_POS) {
    char *s[100];
    int i = 0;

    char *cur_mode = "file_read";
    char *cur_sub_mode = "";

    while (fgets(&s, sizeof(s), SCENE_FILE))
    {
        if (starts_with(s, "//") == 1) {
            i++;
            continue;
        }
        printf("Line %i: %i, %s", i, strcmp(s, MODES[1].mode), s);
        if (strcmp(cur_mode, "file_read") == 0) {
            
        }
        // if (mode == "file_read") {
        //     for (int i = 0; i < READ_MODES; i++) {
        //         if (strcmp())
        //     }
        // }
        i++;
    }

    return;
}
