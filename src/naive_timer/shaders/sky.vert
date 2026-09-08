#version 330 core

// A fullscreen triangle, generated with no vertex buffer at all: three
// vertices at (-1,-1), (3,-1), (-1,3) cover the viewport, and the GPU clips
// the overhang. Cheaper than a quad -- one triangle means no seam down the
// diagonal where two triangles would meet, and no attribute fetch.
//
// Core profile still requires *some* VAO bound, even an empty one.

out vec2 vUV;   // 0..1 across the viewport

void main() {
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUV = corner;
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
