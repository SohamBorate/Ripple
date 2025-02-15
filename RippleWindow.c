#include <windows.h>
#include <stdint.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "BmpWindow.h"
#include "sphere.h"

#define LIGHT_INTENSITY 1

void render(vec3 *origin, vec3 *sun, sphere *ball, int height, int width, RGB_window pixels[height][width]);

float get_point_from_sphere(vec3 *origin, vec3 *direction, sphere *ball);

// Example: 1024x1024 pixel buffer (RGB_window format)
#define WIDTH 256
#define HEIGHT 256
#define PI 3.14

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    const char CLASS_NAME[] = "SampleWindowClass";

    // AllocConsole();                       // Create a console window
    // freopen("CONOUT$", "w", stdout);      // Redirect stdout to the console

    // Define a window class
    WNDCLASS wc = {0};

    RGB_window (*pixels)[WIDTH] = calloc(HEIGHT, WIDTH * sizeof(RGB_window));

    vec3 *vector = vec3_new(0.0,0.0,10.0);
    if (vector == NULL) {return 1;}

    sphere *ball = sphere_new(vector, 4.0);
    if (ball == NULL) {return 2;}

    sphere_print(ball);

    vec3 *origin = vec3_new(0.0,0.0,0.0);
    if (origin == NULL) {
        sphere_free(ball);
        return 5;
    }
    double sun_index = 0.0;
    // Window procedure function to handle events
    LRESULT CALLBACK WindowProcedure(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
        static HDC hdc;
        static PAINTSTRUCT ps;

        switch (uMsg) {
            case WM_CREATE:
                vec3 *sun1 = vec3_new(0.0, 10.0, 0.0);
                render(origin, sun1, ball, HEIGHT, WIDTH, pixels);
                SetTimer(hwnd, 1, 16, NULL);  // Timer with ID 1, 16 ms interval (60 FPS)
                return 0;

            case WM_PAINT:
                hdc = BeginPaint(hwnd, &ps);

                // Create a BITMAPINFO structure for the pixel data
                BITMAPINFO bmi = {0};
                bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
                bmi.bmiHeader.biWidth = WIDTH;
                bmi.bmiHeader.biHeight = HEIGHT;  // Negative to indicate top-down bitmap
                bmi.bmiHeader.biPlanes = 1;
                bmi.bmiHeader.biBitCount = 24;     // 24-bit RGB
                bmi.bmiHeader.biCompression = BI_RGB;

                // Draw the pixel array to the window
                StretchDIBits(
                    hdc,
                    0, 0, WIDTH, HEIGHT,          // Destination rectangle
                    0, 0, WIDTH, HEIGHT,          // Source rectangle
                    pixels,                   // Pointer to the pixel data
                    &bmi,                         // BITMAPINFO structure
                    DIB_RGB_COLORS, SRCCOPY       // Options
                );

                EndPaint(hwnd, &ps);
                return 0;

            case WM_TIMER:
                sun_index += (2.0 * PI) * (16.0 / 1000.0);
                vec3 *sun2 = vec3_new(10.0 * cos(sun_index), 5.0 * sin(sun_index * 3.0), 10 + 10.0 * sin(sun_index));
                // vec3_print(sun2);
                // printf("%d\n", sun_index);
                if (sun_index > (2.0 * PI)) {
                    sun_index -= (2.0 * PI);
                }
                render(origin, sun2, ball, HEIGHT, WIDTH, pixels);

                InvalidateRect(hwnd, NULL, FALSE);
                return 0;
            case WM_CLOSE:
                KillTimer(hwnd, 1);
                PostQuitMessage(0);
                return 0;
            case WM_DESTROY:
                PostQuitMessage(0);
                free(pixels);
                vec3_free(origin);
                sphere_free(ball);
                return 0;
            default:
                return DefWindowProc(hwnd, uMsg, wParam, lParam);
        }
    }

    wc.lpfnWndProc = WindowProcedure;           // Pointer to the window procedure
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    // Register the window class
    RegisterClass(&wc);

    // Create the window
    HWND hwnd = CreateWindowEx(
        WS_EX_CLIENTEDGE,                                  // Optional window styles
        CLASS_NAME,                         // Window class name
        "Ripple",           // Window title
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX, // Window style
        CW_USEDEFAULT, CW_USEDEFAULT,       // Position (x, y)
        WIDTH, HEIGHT,                           // Size (width, height)
        NULL, NULL, hInstance, NULL);       // Parent window, Menu, Instance, Additional data

    if (hwnd == NULL) {
        return 0;
    }

    ShowWindow(hwnd, nCmdShow);

    // Main message loop
    MSG msg = {0};
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}

void render(vec3 *origin, vec3 *sun, sphere *ball, int height, int width, RGB_window pixels[height][width]) {
    const float start_x = 0.5 - (float)((1.0 / width) / 2.0);
    const float start_y = 0.5 - (float)((1.0 / height) / 2.0);
    vec3 *render_pos = vec3_new(origin->x + start_x, origin->y + start_y, origin->z + 0.7);
    // vec3 *sun = vec3_new(0.0,-10.0,0.0);
    printf("Sun position: ");
    vec3_print(sun);

    // render
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            vec3 *direction = vec3_difference(render_pos, origin);

            float point = get_point_from_sphere(origin, direction, ball);

            if (point >= 0) {
                // sphere
                // RGB_window 118, 185, 0
                RGB_window color;
                color.blue = 185;
                color.green = 0;
                color.red = 118;

                vec3 *hit_pos = vec3_scalar_product(direction, point);
                vec3 *nhit_pos = vec3_difference(hit_pos, ball->position);
                vec3 *normal = vec3_scalar_product(nhit_pos, (float)(1.0 / vec3_magnitude(nhit_pos)));
                vec3 *light = vec3_difference(sun, hit_pos);
                float diffusion = (vec3_dot_vec3(normal,light) / (vec3_magnitude(normal) * vec3_magnitude(light)));

                pixels[i][j].blue = validate_rgb((int)(color.blue + (color.blue * diffusion * LIGHT_INTENSITY)));
                pixels[i][j].green = validate_rgb((int)(color.green + (color.green * diffusion * LIGHT_INTENSITY)));
                pixels[i][j].red = validate_rgb((int)(color.red + (color.red * diffusion * LIGHT_INTENSITY)));

                vec3_free(hit_pos);
                vec3_free(nhit_pos);
                vec3_free(normal);
                vec3_free(light);
            } else {
                // background
                // RGB 0, 0, 0
                pixels[i][j].blue = validate_rgb(0);
                pixels[i][j].green = validate_rgb(0);
                pixels[i][j].red = validate_rgb(0);
            }

            vec3_free(direction);
            // move 1 pixel to the right
            render_pos->x = render_pos->x - (float)(1.0 / (width));
        }
        // move 1 pixel to the bottom start
        render_pos->x = origin->x + start_x;
        render_pos->y = render_pos->y - (float)(1.0 / (height));
    }

    vec3_free(render_pos);
    vec3_free(sun);
}

float get_point_from_sphere(vec3 *origin, vec3 *direction, sphere *ball) {
    vec3 *ball_to_origin = vec3_difference(origin, ball->position);
    if (vec3_magnitude(ball_to_origin) <= ball->radius) {return -1;}

    // quadratic equation
    float a = pow(vec3_magnitude(direction), 2);
    float b = 2 * vec3_dot_vec3(direction, ball_to_origin);
    float c = pow(vec3_magnitude(ball_to_origin), 2) - pow(ball->radius, 2);
    float D = pow(b,2) - (4*a*c);

    vec3_free(ball_to_origin);
    if (!(D >= 0)) {return D;}

    // quadratic formula
    float x1 = (((-1 * b) + sqrt(D)) / (2 * a));
    float x2 = (((-1 * b) - sqrt(D)) / (2 * a));

    return fmin(x1, x2);
}

