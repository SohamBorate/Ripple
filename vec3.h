typedef struct {
    float x;
    float y;
    float z;
} vec3;

void vec3_print(vec3 *v);

vec3* vec3_new(float x, float y, float z);

void vec3_free(vec3 *v);

float vec3_magnitude(vec3 *v);

vec3 *vec3_unit(vec3 *v);

vec3* vec3_sum(vec3 *v1, vec3 *v2);

vec3* vec3_difference(vec3 *v1, vec3 *v2);

vec3* vec3_scalar_product(vec3 *v, float s);

vec3* vec3_vec3_product(vec3 *v1, vec3 *v2);

float vec3_dot_scalar(vec3 *v, float s);

float vec3_dot_vec3(vec3 *v1, vec3 *v2);

vec3* vec3_cross_vec3(vec3 *v1, vec3 *v2);