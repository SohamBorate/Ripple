# Directories
SRC_DIR = src
BUILD_DIR = build

# Compiler and flags
CC = cl
CFLAGS = /O2 /Wall

# Source files
SOURCES = $(SRC_DIR)/bmp.c $(SRC_DIR)/ripple.c $(SRC_DIR)/sphere.c $(SRC_DIR)/cube.c $(SRC_DIR)/plane.c $(SRC_DIR)/vec3.c $(SRC_DIR)/render.c $(SRC_DIR)/raycast.c $(SRC_DIR)/string_utils.c $(SRC_DIR)/file_parser.c
OBJECTS = $(BUILD_DIR)/bmp.obj $(BUILD_DIR)/ripple.obj $(BUILD_DIR)/sphere.obj $(BUILD_DIR)/cube.obj $(BUILD_DIR)/plane.obj $(BUILD_DIR)/vec3.obj $(BUILD_DIR)/render.obj $(BUILD_DIR)/raycast.obj  $(BUILD_DIR)/string_utils.obj $(BUILD_DIR)/file_parser.obj

# Output executable
TARGET = $(BUILD_DIR)/ripple.exe

# Build executable
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) /Fe$(TARGET)

# Compile C files
$(BUILD_DIR)/%.obj: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) /c $< /Fo$@

# Clean build directory
clean:
	del /Q $(BUILD_DIR)\*.obj $(BUILD_DIR)\*.exe
