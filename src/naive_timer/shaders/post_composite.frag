#version 330 core

// The one and only tonemap. Everything upstream -- sky, glass, numerals, glare
// -- is linear radiance with no ceiling; this is where it becomes pixels.
//
// It used to live at the bottom of shard.frag, which meant the shard was
// tonemapped and the sky behind it was not, and the glass alpha-blended over
// the backdrop in display space. Moving it here is what makes an intensity
// slider mean anything: a highlight can now carry radiance 20 while the body of
// the shard sits at 0.5, and the curve keeps both.
//
// EDIT ME -- hot-reloaded on save.

in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform sampler2D uFlare;

uniform float uExposure;        // stop adjustment on the whole frame
uniform float uBloomStrength;   // 0 = no glare at all
uniform float uFlareStrength;   // 0 = no ghosts, halo or streaks
uniform float uRolloff;         // shoulder; see the note below

void main() {
    vec3 c = texture(uScene, vUV).rgb * uExposure;

    // Both additive, because glare and flare are light that scattered on its
    // way to the sensor -- added to what was already there, not blended with
    // it. Adding in linear radiance also means a flare landing on an already
    // bright area pushes it further up the tonemap's shoulder, rather than
    // washing it out the way an alpha blend would.
    c += texture(uBloom, vUV).rgb * uBloomStrength;
    c += texture(uFlare, vUV).rgb * uFlareStrength;

    // Per-channel Reinhard. Per-channel is the point, not an oversight: it
    // saturates the strongest channel first, so a hot core converges on white
    // while the dim falloff keeps the light's tint. That is where the
    // "brightness washes to #ffffff" behaviour comes from, and it is why no
    // explicit blend toward white is needed anywhere in this pipeline.
    //
    // The curve reaches 1.0 at radiance 1/(1 - uRolloff) and flat-clips beyond,
    // so uRolloff is the headroom control: raise it alongside light_intensity.
    // At 1.0 and above it asymptotes and never clips at all.
    c = c / (1.0 + c * uRolloff);

    FragColor = vec4(c, 1.0);
}
