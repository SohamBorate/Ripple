#include <math.h>

#include "raycast.h"

// raycast for a sphere, and a sphere only
RaycastResult sphere_raycast(vec3 origin, vec3 direction, int object_index, sphere ball) {
    RaycastResult result;

    direction = vec3_unit(direction);

    vec3 d_vec3 = vec3_difference(ball.position, origin);
    float d_mag = vec3_magnitude(d_vec3);

    // perpendicular distance
    float cos_theta = vec3_dot_vec3(d_vec3, direction) / d_mag;
    const float perp_distance = (float)sqrt(pow(d_mag, 2) * (1 - pow(cos_theta, 2)));

    // does not hit
    if (perp_distance > ball.radius) {
        result.BasePartIndex = -1;
        return result;
    }

    const float hit_distance = d_mag * cos_theta - (float)sqrt(pow(ball.radius, 2) - pow(perp_distance, 2));

    result.BasePartIndex = object_index;
    result.hit_position = vec3_scalar_product(direction, hit_distance);
    result.hit_distance = hit_distance;
    result.hit_normal = vec3_difference(result.hit_position, ball.position);
    result.color = ball.color;

    return result;
}

RaycastResult plane_raycast(vec3 origin, vec3 direction, int object_index, plane plane) {
    RaycastResult result;

    if (vec3_magnitude(direction) != 1.0) {
        direction = vec3_unit(direction);
    }

    float cos_theta = vec3_dot_vec3(vec3_scalar_product(direction, -1.0), vec3_unit(plane.normal));

    if (cos_theta > 1.0 || cos_theta <= 0.0) {
        result.BasePartIndex = -1;
        return result;
    }

    // ax + by + cz + d = 0
    float d = -1.0 * (vec3_dot_vec3(plane.normal, plane.point));
    float perp_distance = (vec3_dot_vec3(origin, plane.normal) + d) / vec3_magnitude(plane.normal);

    if (perp_distance < 0.0) {
        result.BasePartIndex = -1;
        return result;
    }

    const float hit_distance = perp_distance / cos_theta;

    result.BasePartIndex = object_index;
    result.hit_position = vec3_scalar_product(direction, hit_distance);
    result.hit_distance = hit_distance;
    result.hit_normal = plane.normal;
    result.color = plane.color;

    return result;
}

RaycastResult raycast(vec3 origin, vec3 direction, int num_objects, BasePart *objects, int ignore_object_index) {
    RaycastResult result;

    result.BasePartIndex = -1;
    float hit_distance = 10000000000;

    if (vec3_magnitude(direction) != 1.0) {
        direction = vec3_unit(direction);
    }

    for (int i = 0; i < num_objects; i++) {
        if (i == ignore_object_index) {
            continue;
        }
        if (objects[i].DataType == DATA_TYPE_SPHERE) {
            sphere ball = objects[i].sphere;

            RaycastResult temp_result = sphere_raycast(origin, direction, i, ball);

            if (temp_result.BasePartIndex != -1) {
                if (temp_result.hit_distance < hit_distance && temp_result.hit_distance > 0) {
                    hit_distance = temp_result.hit_distance;
                    result = temp_result;
                }
            }
        } else if (objects[i].DataType == DATA_TYPE_PLANE) {
            plane plane = objects[i].plane;

            RaycastResult temp_result = plane_raycast(origin, direction, i, plane);

            if (temp_result.BasePartIndex != -1) {
                if (temp_result.hit_distance < hit_distance && temp_result.hit_distance > 0) {
                    hit_distance = temp_result.hit_distance;
                    result = temp_result;
                }
            }
        }
    }

    return result;
}