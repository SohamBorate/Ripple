# Directories
SRC_DIR = src
BUILD_DIR = build

# Compiler and flags
NVCC = nvcc
CC = cl
CFLAGS = /O2 /Wall
NVFLAGS = -arch=sm_75 -gencode arch=compute_75,code=sm_75 -O2

# Source files
SOURCES_C = $(SRC_DIR)/bmp.c $(SRC_DIR)/ripple.c $(SRC_DIR)/sphere.c $(SRC_DIR)/vec3.c
SOURCES_CU = $(SRC_DIR)/render.cu $(SRC_DIR)/sphere_cuda.cu $(SRC_DIR)/vec3_cuda.cu
OBJECTS = $(BUILD_DIR)/bmp.obj $(BUILD_DIR)/ripple.obj $(BUILD_DIR)/sphere.obj $(BUILD_DIR)/vec3.obj \
          $(BUILD_DIR)/render.obj $(BUILD_DIR)/sphere_cuda.obj $(BUILD_DIR)/vec3_cuda.obj $(BUILD_DIR)/cuda_link.obj

# Output executable
TARGET = $(BUILD_DIR)/ripple.exe

# Build executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVFLAGS) $(OBJECTS) -o $(TARGET) -lcudart

# Compile C files
$(BUILD_DIR)/%.obj: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) /c $< /Fo$@

# Compile CUDA files (device code compilation with -dc)
$(BUILD_DIR)/%.obj: $(SRC_DIR)/%.cu
	$(NVCC) $(NVFLAGS) -dc -c $< -o $@

# CUDA linking stage
$(BUILD_DIR)/cuda_link.obj: $(BUILD_DIR)/render.obj $(BUILD_DIR)/sphere_cuda.obj $(BUILD_DIR)/vec3_cuda.obj
	$(NVCC) $(NVFLAGS) -dlink $^ -o $@

# Clean build directory
clean:
	del /Q $(BUILD_DIR)\*.obj $(BUILD_DIR)\*.exe $(BUILD_DIR)\*.exp $(BUILD_DIR)\*.lib
