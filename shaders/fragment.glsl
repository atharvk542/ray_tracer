#version 330

#define TRIANGLE_MAX 30
uniform int tri_count;
uniform vec3 iResolution;
uniform vec3 camera_position;
uniform vec3 camera_direction;

// A triangle along with some shading parameters
struct triangle_t {
    vec4 positions[3];
    vec4 normal;
    vec4 color;
    vec4 emission;

    float reflectivity;
    float transparency;
    float ior;
};

layout(std140) uniform TriangleBuffer {
    triangle_t tris[TRIANGLE_MAX];
};

out vec4 fragColor;

// Computes the direction vector of a primary ray
// \param x, y Coordinates between 0 and 1 for the pixel that the ray
//      passes through (x, y)
// \param camera_position Position vector for the camera (o)
// \param left_bottom Position vector for the left bottom corner of
//      the image plane (c)
// \param right, up Direction vectors towards the right and top end
//      of the image plane (r, u)
// \return A direction vector of length 1 for a ray through the
//      specified pixel (d)
vec3 get_primary_ray_direction(float x, float y, vec3 camera_position, vec3 left_bottom, vec3 right, vec3 up) {
    vec3 p = left_bottom + x * right + y * up;
    return normalize(p - camera_position);
}

// Checks whether a ray intersects a triangle
// \param out_t The ray parameter at the intersection (if any) (t)
// \param origin The position at which the ray starts (o)
// \param direction The direction vector of the ray (d)
// \param tri The triangle for which to check an intersection
// \return true if there is an intersection, false otherwise
bool ray_triangle_intersection(out float out_t, vec3 origin, vec3 direction, triangle_t tri) {
    out_t = 0.0;

    vec3 e_1 = tri.positions[1].xyz - tri.positions[0].xyz;
    vec3 e_2 = tri.positions[2].xyz - tri.positions[0].xyz;

    mat3 cols = mat3(-direction, e_1, e_2);
    vec3 pos_minus_v0 = origin - tri.positions[0].xyz;
    vec3 cof = inverse(cols) * pos_minus_v0;

    out_t = cof[0];

    if (cof[0] >= float(0) && cof[1] >= float(0) && cof[2] >= float(0) && cof[1] + cof[2] <= float(1)) {
        return true;
    }

    return false;
}

vec3 getCenter(triangle_t tri) {
    return (tri.positions[0].xyz + tri.positions[1].xyz + tri.positions[2].xyz) / 3.0;
}

// Checks whether a ray intersects any triangle of the mesh
// \param out_t The ray parameter at the closest intersection (if any) (t)
// \param out_index The index of the closest triangle that was intersected (if any), -1 otherwise
// \param origin The position at which the ray starts (o)
// \param direction The direction vector of the ray (d)
// \return true if there is an intersection, false otherwise
bool ray_mesh_intersection(
    out float out_t,
    out int out_index,
    out triangle_t light_sources[TRIANGLE_MAX],
    out int ind,
    vec3 origin,
    vec3 direction
) {
    // find the nearest intersection across all triangles
    out_t = 1.0e38;

    bool hit = false;
    int hit_index = -1;

    ind = 0;

    for (int i = 0; i < tri_count; i++) {
        float temp = 0.0;

        // track if the triangle is a light source for later lighting detection
        if (length(tris[i].emission.xyz) > 0.0) {
            light_sources[ind] = tris[i];
            ind++;
        }

        // check for intersection with each triangle, return the closest one
        if (ray_triangle_intersection(temp, origin, direction, tris[i])) {
            if (temp < out_t) {
                out_t = temp;
                hit_index = i;
                hit = true;
            }
        }
    }

    out_index = hit_index;
    return hit;
}

// Check whether point is in shadow with respect to a light source
bool isInShadow(
    vec3 hit_point,
    vec3 normal,
    vec3 light_position
) {
    const float shadow_bias = 0.001;

    // offset shadow ray along the hit normal by a little bit to avoid self intersection
    vec3 shadow_origin = hit_point + normal * shadow_bias;

    // calculate direction and distance from the new origin.
    vec3 toLight = light_position - shadow_origin;
    float max_distance = length(toLight);
    vec3 light_dir = normalize(toLight);

    float shadow_t;
    int shadow_index;
    int dummy_ind;

    triangle_t dummy_lights[TRIANGLE_MAX];

    if (ray_mesh_intersection(
        shadow_t,
        shadow_index,
        dummy_lights,
        dummy_ind,
        shadow_origin,
        light_dir
    )) {
        return shadow_t < max_distance - 1e-4;
    }

    return false;
}

// calculate main lighting at a ray intersection point with lambertian shading, ambient light, and shadows
vec3 calculateLighting(
    triangle_t tri,
    vec3 hit_point,
    vec3 ray_direction,
    triangle_t light_sources[TRIANGLE_MAX],
    int ind
) {
    float PI = 3.14159265;

    if (length(tri.emission.xyz) > 0.0)
        return tri.emission.xyz;

    vec3 lighting = 0.1 * tri.color.xyz;

    // use triangle's surface normal for lighting and shadows (it breaks if u use light direciton)
    vec3 normal = normalize(tri.normal.xyz);

    for (int i = 0; i < ind; i++) {
        vec3 light_center = getCenter(light_sources[i]);
        vec3 light_dir = normalize(light_center - hit_point);

        float NdotL = max(0.0, dot(normal, light_dir));

        if (NdotL > 0.0 &&
            !isInShadow(hit_point, normal, light_center)) {

            lighting +=
                (tri.color.xyz / PI) *
                light_sources[i].emission.xyz *
                NdotL / float(ind);
        }
    }

    return lighting;
}

vec3 traceRay(vec3 ray_origin, vec3 ray_direction) {

    vec3 result = vec3(0.0);
    vec3 throughput = vec3(1.0);

    // trace ray through multiple bounces in scene for reflections and refractions
    for (int bounce = 0; bounce < 3; bounce++) {

        float t;
        int tri_index;
        int ind;

        triangle_t light_sources[TRIANGLE_MAX];

        if (!ray_mesh_intersection(
            t, tri_index, light_sources, ind,
            ray_origin, ray_direction
        ))
            break;

        triangle_t tri = tris[tri_index];

        vec3 hit_point = ray_origin + t * ray_direction;
        vec3 normal = normalize(tri.normal.xyz);

        // calculate lighting at the intersection for each bounce
        result += throughput * calculateLighting(
            tri, hit_point, ray_direction, light_sources, ind
        );

        // stop if surface isn't reflective or transparent
        if (tri.reflectivity <= 0.0 && tri.transparency <= 0.0)
            break;

        // calculate next ray direction
        if (tri.reflectivity > tri.transparency) {
            ray_direction = reflect(ray_direction, normal);
            throughput *= tri.reflectivity;
        }
        else {
            ray_direction = refract(
                ray_direction,
                normal,
                1.0 / tri.ior
            );

            throughput *= tri.transparency;
        }

        ray_origin = hit_point + ray_direction * 0.001;
    }

    return result;
}

void mainImage(out vec4 out_color, in vec2 pixel_coord) {
    // control image plane with camera orientation (GPT code)
    // Camera orientation
    vec3 forward = normalize(camera_direction);

    vec3 world_up = vec3(0.0, 0.0, 1.0);

    // Camera coordinate system
    vec3 right = normalize(cross(forward, world_up));
    vec3 up = normalize(cross(right, forward));

    // Image plane settings
    float aspect = iResolution.x / iResolution.y;

    float fov = radians(60.0);

    float half_height = tan(fov * 0.5);
    float half_width = aspect * half_height;

    // Image plane center
    vec3 middle = camera_position + forward;

    // Scale image plane
    right *= 2.0 * half_width;
    up *= 2.0 * half_height;

    // Bottom-left image plane corner
    vec3 left_bottom =
        middle - 0.5 * right - 0.5 * up;

    // Compute the camera ray
    vec2 tex_coord = pixel_coord / iResolution.xy;
    vec3 ray_direction = get_primary_ray_direction(
        tex_coord.x, tex_coord.y, camera_position, left_bottom, right, up);

    // Trace a ray and display the triangle color+emission
    out_color.rgb = traceRay(camera_position, ray_direction);
    out_color.a = 1.0;
}

void main() {
    mainImage(fragColor, gl_FragCoord.xy);
}