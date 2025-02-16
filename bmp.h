#include <stdint.h>

typedef struct {
    uint16_t signature;     // File signature "BM" (0x4D42)
    uint32_t fileSize;      // Size of the BMP file
    uint16_t reserved1;     // Reserved (unused)
    uint16_t reserved2;     // Reserved (unused)
    uint32_t offset;        // Offset to the start of image data
    uint32_t headerSize;    // Size of the header (40 bytes)
    int32_t width;          // Width of the image
    int32_t height;         // Height of the image
    uint16_t planes;        // Number of color planes (must be 1)
    uint16_t bitsPerPixel;  // Number of bits per pixel (usually 24)
    uint32_t compression;   // Compression method (usually 0 for uncompressed)
    uint32_t imageSize;     // Size of the image data (may be 0 for uncompressed)
    int32_t xPixelsPerMeter;// Horizontal pixels per meter
    int32_t yPixelsPerMeter;// Vertical pixels per meter
    uint32_t colorsUsed;    // Number of colors used in the bitmap
    uint32_t colorsImportant; // Number of important colors
} __attribute__((__packed__))
BMPHeader;

typedef struct
{
    uint8_t  blue;
    uint8_t  green;
    uint8_t  red;
} __attribute__((__packed__))
RGB;

int write_bmp(const char *filename, const int width, const int height, RGB *pixels);

uint8_t validate_rgb(int n);