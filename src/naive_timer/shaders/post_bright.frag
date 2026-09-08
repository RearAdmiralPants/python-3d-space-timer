#version 330 core

// Bright-pass: what the glare is made of.
//
// Runs at half resolution, reading the resolved HDR scene. This is the pass
// that answers "where is the light landing particularly focused" -- and the
// answer needs no edge detection, no curvature term, and no knowledge of the
// geometry. The bevel's specular lobe and the Fresnel rim are already the
// brightest pixels in the frame by a wide margin, so a luminance threshold
// finds them on its own. That is the whole trick.
//
// It only works because the scene buffer is floating point. In an 8-bit buffer
// every one of those pixels has already been flattened to the same white, and
// thresholding tells you *that* they are bright but not *how* bright -- a
// searing highlight and a merely bright one would emit identical glare.
//
// EDIT ME -- hot-reloaded on save, like the rest.

in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uScene;
uniform vec2 uTexel;        // 1 / full-resolution size, for the box tap
uniform float uExposure;
uniform float uThreshold;   // radiance below this contributes no glare

void main() {
    // Four taps at the full-res texel corners, i.e. an explicit 2x2 box. A
    // single tap at half res would alias the highlight: a specular lobe on a
    // curved bevel is small and moves, so point-sampling it makes the glare
    // crawl and flicker as the camera sways. Averaging first is what stops the
    // fireflies.
    vec3 sum =
          texture(uScene, vUV + vec2(-0.5, -0.5) * uTexel).rgb
        + texture(uScene, vUV + vec2( 0.5, -0.5) * uTexel).rgb
        + texture(uScene, vUV + vec2(-0.5,  0.5) * uTexel).rgb
        + texture(uScene, vUV + vec2( 0.5,  0.5) * uTexel).rgb;
    vec3 c = sum * 0.25 * uExposure;

    // Soft knee. A hard step at the threshold makes the glare pop in and out
    // as a highlight drifts across it; scaling by the *fraction* of luminance
    // that clears the bar fades it in instead.
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    float over = max(luma - uThreshold, 0.0);
    FragColor = vec4(c * (over / max(luma, 1e-4)), 1.0);
}
