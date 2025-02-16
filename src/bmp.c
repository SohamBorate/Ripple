#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bmp.h"

int write_bmp(const char *filename, const int WIDTH, const int HEIGHT, RGB *pixels) {
    FILE *file = fopen(filename, "wb"); // Open the BMP file in binary write mode

    if (file == NULL) {
        printf("Memory allocation for Bitmap failed\n");
        return 8;
    }

    // BMP header setup
    BMPHeader header;
    header.signature = 0x4D42;
    header.fileSize = sizeof(BMPHeader) + WIDTH * HEIGHT * 3; // 3 bytes per pixel (24-bit)
    header.reserved1 = 0;
    header.reserved2 = 0;
    header.offset = sizeof(BMPHeader);
    header.headerSize = 40;
    header.width = WIDTH;
    header.height = HEIGHT;
    header.planes = 1;
    header.bitsPerPixel = 24;
    header.compression = 0;
    header.imageSize = WIDTH * HEIGHT * 3;
    header.xPixelsPerMeter = 0;
    header.yPixelsPerMeter = 0;
    header.colorsUsed = 0;
    header.colorsImportant = 0;

    const int padding = (4 - (header.width * sizeof(RGB)) % 4) % 4;

    // Write BMP header
    fwrite(&header, sizeof(header), 1, file);

    // Write new pixels to outfile
    for (int i = HEIGHT - 1; i >= 0; i--) {
        // Write row to outfile
        fwrite(pixels + i * WIDTH, sizeof(RGB), WIDTH, file);

        // Write padding at end of row
        for (int k = 0; k < padding; k++)
        {
            fputc(0x00, file);
        }
    }

    fclose(file);

    return 0;
}
