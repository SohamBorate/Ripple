#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "string_utils.h"

int starts_with(const char *text, const char *match) {
    const int len1 = strlen(text);
    const int len2 = strlen(match);

    if (len2 > len1) {
        return 0;
    } else if (len2 == len1) {
        if (strcmp(text, match) == 0) {
            return 1;
        } else {
            return 0;
        }
    } else {
        int status = 1;
        for (int i = 0; i < len2; i++) {
            if (text[i] != match[i]) {
                status = 0;
                break;
            }
        }
        return status;
    }
}

int ends_with(const char *text, const char *match) {
    const int len1 = strlen(text);
    const int len2 = strlen(match);

    if (len2 > len1) {
        return 0;
    } else if (len2 == len1) {
        if (strcmp(text, match) == 0) {
            return 1;
        } else {
            return 0;
        }
    } else {
        int status = 1;
        int t = len1 - len2;
        for (int i = 0; i < len2; i++) {
            if (text[t] != match[i]) {
                status = 0;
                break;
            }
            t++;
        }
        return status;
    }
}
