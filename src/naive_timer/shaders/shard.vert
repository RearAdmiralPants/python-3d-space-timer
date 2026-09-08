#version 330 core

// Faceted glass shard. Each triangle is a flat-shaded facet, so the specular
// highlight breaks across the bevels instead of sliding smoothly over a dome.
//
// On shatter, each wedge becomes an independent rigid body: it spins about its
// own centroid and translates away under its own velocity, rather than being
// pushed radially outward from the shard's centre (which read as a pinwheel).

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;

// Per-wedge rigid-body state. Every vertex of a wedge carries identical
// values, so the whole solid chunk moves together.
layout(location = 3) in vec3 aPieceCenter;  // pivot: the wedge's centroid
layout(location = 4) in vec3 aPieceVel;     // linear velocity, units/sec
layout(location = 5) in vec3 aPieceAxis;    // rotation axis; |axis| = rad/sec
layout(location = 6) in float aCap;         // 1.0 on the radial cut faces

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform mat3 uNormalMat;

uniform float uShatterT;     // seconds since the break; 0 while intact
uniform float uGravity;      // downward accel of the falling pieces (scaled g)
uniform float uSpin;         // radians, slow idle rotation
uniform float uSpinAtBreak;  // uSpin sampled the instant the shard broke

out vec3 vWorld;
out vec3 vNormal;
out vec2 vUV;
out float vCap;

// Gravity, so the pieces fall away rather than drifting flat. Magnitude comes
// in as a uniform (uGravity); 0 leaves them drifting on their launch velocity.

mat3 axisRotation(vec3 axis, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    float t = 1.0 - c;
    vec3 a = normalize(axis);
    return mat3(
        t * a.x * a.x + c,       t * a.x * a.y + s * a.z, t * a.x * a.z - s * a.y,
        t * a.x * a.y - s * a.z, t * a.y * a.y + c,       t * a.y * a.z + s * a.x,
        t * a.x * a.z + s * a.y, t * a.y * a.z - s * a.x, t * a.z * a.z + c
    );
}

mat3 spinY(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat3(c, 0.0, -s,
                0.0, 1.0, 0.0,
                s, 0.0, c);
}

void main() {
    // The shard breaks from wherever its idle rotation had got to. Freezing
    // the angle at the break, rather than dropping back to zero, is what stops
    // the shard snapping to its start pose one frame before it shatters.
    mat3 spin = spinY(uShatterT > 0.0 ? uSpinAtBreak : uSpin);

    vec3 pos = spin * aPos;
    vec3 nrm = spin * aNormal;

    if (uShatterT > 0.0) {
        float t = uShatterT;

        // The rigid-body state was authored in the shard's rest frame, so it
        // has to be carried into the frozen pose along with the geometry.
        vec3 centre = spin * aPieceCenter;
        vec3 vel = spin * aPieceVel;
        vec3 axis = spin * aPieceAxis;

        float rate = length(axis);
        mat3 tumble = rate > 1e-5 ? axisRotation(axis, rate * t) : mat3(1.0);

        // Rotate about the wedge's own centroid, then carry it away.
        pos = centre + tumble * (pos - centre);
        pos += vel * t + 0.5 * vec3(0.0, -uGravity, 0.0) * t * t;
        nrm = tumble * nrm;
    }

    vec4 world = uModel * vec4(pos, 1.0);
    vWorld = world.xyz;
    vNormal = normalize(uNormalMat * nrm);
    vUV = aUV;
    vCap = aCap;

    gl_Position = uProj * uView * world;
}
