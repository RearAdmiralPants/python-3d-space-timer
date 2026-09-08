#version 330 core

// One half of a separable Gaussian. The caller runs it twice per iteration --
// once with uDir horizontal, once vertical -- which turns an NxN kernel into
// 2N taps instead of N^2.
//
// Called repeatedly with a growing uDir rather than with one huge kernel: three
// iterations at steps 1, 2 and 4 texels give a far wider, softer falloff than
// nine taps ever could, for 6 cheap passes over a half-resolution buffer.
//
// EDIT ME -- hot-reloaded on save.

in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uSource;
uniform vec2 uDir;   // one step along the blur axis, in UV units

// sigma ~= 2 texels, normalised so the weights sum to 1 and the blur neither
// brightens nor dims what it spreads.
const float W[5] = float[](0.227027, 0.194595, 0.121622, 0.054054, 0.016216);

void main() {
    vec3 c = texture(uSource, vUV).rgb * W[0];
    for (int i = 1; i < 5; i++) {
        vec2 offset = uDir * float(i);
        c += texture(uSource, vUV + offset).rgb * W[i];
        c += texture(uSource, vUV - offset).rgb * W[i];
    }
    FragColor = vec4(c, 1.0);
}
