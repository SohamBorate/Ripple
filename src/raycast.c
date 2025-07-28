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
    result.hit_position = vec3_sum(origin,
        vec3_scalar_product(direction, hit_distance)
    );

    result.hit_distance = hit_distance;
    result.hit_normal = vec3_unit(vec3_difference(result.hit_position, ball.position));
    result.color = ball.color;

    return result;
}

RaycastResult plane_raycast(vec3 origin, vec3 direction, int object_index, plane plane) {
    RaycastResult result;

    if (vec3_magnitude(direction) != 1.0) {
        direction = vec3_unit(direction);
    }

    float cos_theta = vec3_dot_vec3(vec3_scalar_product(direction, -1.0), vec3_unit(plane.normal));

    if (cos_theta > 1.0) {
        // cos_theta *= -1.0;
        result.BasePartIndex = -1;
        return result;
    }

    if (cos_theta < 0.0) {
        // cos_theta *= -1.0;
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
    result.hit_position = vec3_sum(origin,
        vec3_scalar_product(direction, hit_distance)
    );

    result.hit_distance = hit_distance;
    result.hit_normal = vec3_unit(plane.normal);
    result.color = plane.color;

    return result;
}

RaycastResult cube_raycast(vec3 origin, vec3 direction, int object_index, cube cube) {
    // front
    // back
    // top
    // bottom
    // right
    // left
    RaycastResult raycast_surfaces[6];

    raycast_surfaces[0] = plane_raycast(origin, direction, object_index, cube.surfaces.front);
    raycast_surfaces[1] = plane_raycast(origin, direction, object_index, cube.surfaces.back);
    raycast_surfaces[2] = plane_raycast(origin, direction, object_index, cube.surfaces.top);
    raycast_surfaces[3] = plane_raycast(origin, direction, object_index, cube.surfaces.bottom);
    raycast_surfaces[4] = plane_raycast(origin, direction, object_index, cube.surfaces.right);
    raycast_surfaces[5] = plane_raycast(origin, direction, object_index, cube.surfaces.left);

    RaycastResult result;

    result.BasePartIndex = -1;
    float hit_distance = 10000000000;

    // validate that hit_position is within bounds
    vec3 hit_position;

    // front - XY plane
    if (raycast_surfaces[0].BasePartIndex != -1) {
        hit_position = raycast_surfaces[0].hit_position;
        if (hit_position.x > cube.world_vertex_1.x || hit_position.x < cube.world_vertex_2.x) {
            raycast_surfaces[0].BasePartIndex = -1;
        } else if (hit_position.y > cube.world_vertex_1.y || hit_position.y < cube.world_vertex_2.y) {
            raycast_surfaces[0].BasePartIndex = -1;
        }
    }

    // back - XY plane
    if (raycast_surfaces[1].BasePartIndex != -1) {
        hit_position = raycast_surfaces[1].hit_position;
        if (hit_position.x > cube.world_vertex_1.x || hit_position.x < cube.world_vertex_2.x) {
            raycast_surfaces[1].BasePartIndex = -1;
        } else if (hit_position.y > cube.world_vertex_1.y || hit_position.y < cube.world_vertex_2.y) {
            raycast_surfaces[1].BasePartIndex = -1;
        }
    }

    // top - XZ plane
    if (raycast_surfaces[2].BasePartIndex != -1) {
        hit_position = raycast_surfaces[2].hit_position;
        if (hit_position.x > cube.world_vertex_1.x || hit_position.x < cube.world_vertex_2.x) {
            raycast_surfaces[2].BasePartIndex = -1;
        } else if (hit_position.z > cube.world_vertex_1.z || hit_position.z < cube.world_vertex_2.z) {
            raycast_surfaces[2].BasePartIndex = -1;
        }
    }

    // bottom - XZ plane
    if (raycast_surfaces[3].BasePartIndex != -1) {
        hit_position = raycast_surfaces[3].hit_position;
        if (hit_position.x > cube.world_vertex_1.x || hit_position.x < cube.world_vertex_2.x) {
            raycast_surfaces[3].BasePartIndex = -1;
        } else if (hit_position.z > cube.world_vertex_1.z || hit_position.z < cube.world_vertex_2.z) {
            raycast_surfaces[3].BasePartIndex = -1;
        }
    }

    // right - YZ plane
    if (raycast_surfaces[4].BasePartIndex != -1) {
        hit_position = raycast_surfaces[4].hit_position;
        if (hit_position.z > cube.world_vertex_1.z || hit_position.z < cube.world_vertex_2.z) {
            raycast_surfaces[4].BasePartIndex = -1;
        } else if (hit_position.y > cube.world_vertex_1.y || hit_position.y < cube.world_vertex_2.y) {
            raycast_surfaces[4].BasePartIndex = -1;
        }
    }

    // left - YZ plane
    if (raycast_surfaces[5].BasePartIndex != -1) {
        hit_position = raycast_surfaces[5].hit_position;
        if (hit_position.z > cube.world_vertex_1.z || hit_position.z < cube.world_vertex_2.z) {
            raycast_surfaces[5].BasePartIndex = -1;
        } else if (hit_position.y > cube.world_vertex_1.y || hit_position.y < cube.world_vertex_2.y) {
            raycast_surfaces[5].BasePartIndex = -1;
        }
    }

    // send results back
    for (int i = 0; i < 6; i++) {
        if (raycast_surfaces[i].BasePartIndex != -1) {
            if (raycast_surfaces[i].hit_distance < hit_distance && raycast_surfaces[i].hit_distance > 0) {
                hit_distance = raycast_surfaces[i].hit_distance;
                result = raycast_surfaces[i];
            }
        }
    }

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
        } else if (objects[i].DataType == DATA_TYPE_CUBE) {
            cube cube = objects[i].cube;

            RaycastResult temp_result = cube_raycast(origin, direction, i, cube);

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