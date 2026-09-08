#version 330 core

// Procedural starfield and nebulae, evaluated in WORLD SPACE along a per-pixel
// view ray. This is a skybox, not a wallpaper: the stars and clouds live on the
// celestial sphere, so an orbiting camera sweeps across them.
//
// They sit at effectively infinite distance, so they rotate with the view
// direction but never translate -- which is exactly how a real starfield
// behaves, and why there is no parallax between them and the shard.
//
// EDIT ME -- hot-reloaded on save, like shard.frag. A failed compile prints
// the error and keeps the last good program.

in vec2 vUV;
out vec4 FragColor;

// Camera basis, world space. The ray is rebuilt from these each pixel.
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamForward;
uniform float uTanHalfFov;
uniform float uAspect;

uniform float uTime;

// The baked nebula. Unused when SKY_PROCEDURAL_NEBULA is defined.
uniform samplerCube uNebulaCube;

uniform float uNebula;        // 0 = empty space, 1 = thick cloud
uniform vec3 uNebulaColorA;   // the cool lobe
uniform vec3 uNebulaColorB;   // the warm lobe
uniform float uStarDensity;   // cells per unit of direction; more = more stars
uniform float uStarBrightness;

// The void is not pure black; a hint of blue reads as depth rather than as an
// unlit buffer.
const vec3 VOID_COLOR = vec3(0.020, 0.024, 0.038);

float hash31(vec3 p) {
    p = fract(p * vec3(127.1, 311.7, 74.7));
    p += dot(p, p.yzx + 34.53);
    return fract((p.x + p.y) * p.z);
}

vec3 hash33(vec3 p) {
    float a = hash31(p);
    float b = hash31(p + a + 1.7);
    return vec3(a, b, hash31(p + b + 5.3));
}

// 3D value noise: smooth interpolation between per-lattice-point randoms.
float noise3(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);          // smoothstep, kills lattice creases

    float n000 = hash31(i + vec3(0.0, 0.0, 0.0));
    float n100 = hash31(i + vec3(1.0, 0.0, 0.0));
    float n010 = hash31(i + vec3(0.0, 1.0, 0.0));
    float n110 = hash31(i + vec3(1.0, 1.0, 0.0));
    float n001 = hash31(i + vec3(0.0, 0.0, 1.0));
    float n101 = hash31(i + vec3(1.0, 0.0, 1.0));
    float n011 = hash31(i + vec3(0.0, 1.0, 1.0));
    float n111 = hash31(i + vec3(1.0, 1.0, 1.0));

    return mix(
        mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
        mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y),
        f.z
    );
}

// Octave count. Overridable at compile time so tools/bench_sky.py can price
// each octave; the app always gets 5.
#ifndef SKY_FBM_OCTAVES
#define SKY_FBM_OCTAVES 5
#endif

float fbm3(vec3 p) {
    float sum = 0.0;
    float amp = 0.5;
    for (int i = 0; i < SKY_FBM_OCTAVES; i++) {
        sum += amp * noise3(p);
        p *= 2.03;                         // not exactly 2: avoids axis banding
        amp *= 0.5;
    }
    return sum;
}

// The nebula, reduced to two scalars that depend only on direction:
//
//    .x  how thick the cloud is here, before uNebula scales it
//    .y  where between the cool and warm lobes its colour sits
//
// Everything colour-dependent stays outside, so uNebula and the two lobe
// colours remain live uniforms -- dragging those sliders needs no re-bake.
// Only the shape is baked, and the shape is what costs 80-95% of the frame.
//
// Defining SKY_PROCEDURAL_NEBULA evaluates the noise directly instead. That is
// the pre-bake path: the bake itself compiles with it, and tools/bench_sky.py
// uses it to price what the bake saves.
vec2 nebulaFactors(vec3 dir) {
#ifdef SKY_PROCEDURAL_NEBULA
    // A second fbm displaces the first (domain warping), which is what turns
    // round blobs into filaments and wisps. The slow drift is the clouds
    // themselves moving, not the camera.
    //
    // uTime is deliberately absent once baked: a snapshot cannot drift. The
    // drift was 0.004 units/s, so this costs little that the eye can follow.
    vec3 drift = vec3(uTime * 0.004, uTime * 0.0015, uTime * 0.002);
    float base = fbm3(dir * 2.3 + drift);
    float warped = fbm3(dir * 3.1 + base * 1.6 - drift);

    // 3D fbm covers far more of the frame than the 2D version did, so the
    // threshold has to climb or the nebula becomes a wash rather than wisps.
    return vec2(smoothstep(0.40, 0.88, warped), smoothstep(0.25, 0.75, base));
#else
    return texture(uNebulaCube, dir).rg;
#endif
}

// One layer of stars on the celestial sphere. Direction space is diced into
// cells; each holds at most one star, placed and lit by the cell's hash.
// Bright stars are made rare by raising a uniform random to a high power.
float starLayer(vec3 dir, float density, float radius, float seed) {
    vec3 grid = dir * density + seed;
    vec3 cell = floor(grid);
    vec3 local = fract(grid);

    vec3 offset = hash33(cell) * 0.7 + 0.15;
    float brightness = pow(hash31(cell + 13.7), 7.0);

    float d = length(local - offset);
    float core = smoothstep(radius, 0.0, d);

    // Twinkle, desynchronised per star.
    float phase = hash31(cell + 3.1) * 6.2831853;
    float twinkle = 0.78 + 0.22 * sin(uTime * 1.7 + phase);

    return core * brightness * twinkle;
}

void main() {
    // Rebuild the world-space view ray for this pixel. Aspect correction lives
    // here, so stars stay round when the window is resized.
    vec2 ndc = vUV * 2.0 - 1.0;
    vec3 dir = normalize(
        uCamForward
        + uCamRight * (ndc.x * uTanHalfFov * uAspect)
        + uCamUp * (ndc.y * uTanHalfFov)
    );

#ifdef SKY_BAKE
    // Baking a cube face: write the two direction-only factors and stop. The
    // camera basis uniforms are the face's axes, at a 90 degree FOV, so this
    // pass walks exactly the directions the runtime lookup will ask for.
    FragColor = vec4(nebulaFactors(dir), 0.0, 1.0);
    return;
#endif

    vec3 color = VOID_COLOR;

    // SKY_SKIP_NEBULA / SKY_SKIP_STARS are never defined by the app. They exist
    // so tools/bench_sky.py can compile half-shaders and attribute the frame
    // cost to the nebula or the stars separately.
    float density = 0.0;
    vec3 cloud = vec3(0.0);

#ifndef SKY_SKIP_NEBULA
    vec2 neb = nebulaFactors(dir);
    density = neb.x * uNebula;
    cloud = mix(uNebulaColorA, uNebulaColorB, neb.y);
    color += cloud * density;
#endif

    // Three star layers at different scales read as depth.
    //
    // The radii are in cell units, and a cell now subtends roughly
    // 1/uStarDensity radians. Too small and a star lands inside a single pixel
    // and flickers out of existence; these keep the core a couple of pixels
    // across at the default density.
    float stars = 0.0;
#ifndef SKY_SKIP_STARS
    stars += starLayer(dir, uStarDensity * 0.6, 0.170, 0.0) * 1.00;
    stars += starLayer(dir, uStarDensity * 1.3, 0.130, 17.0) * 0.65;
    stars += starLayer(dir, uStarDensity * 2.4, 0.095, 41.0) * 0.40;
#endif
    color += vec3(stars) * uStarBrightness;

    // The nebula sits in front of the faintest stars, as dust does.
    color = mix(color, color * 0.75 + cloud * density * 0.4, density * 0.5);

    FragColor = vec4(color, 1.0);
}
