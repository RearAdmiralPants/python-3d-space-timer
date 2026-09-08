#version 330 core

// Lens flare: ghosts, halo and streaks, all read out of the bright-pass buffer.
//
// This is a *camera* artefact, not a lighting one. Nothing here knows where the
// light is, where the bevels are, or which way anything faces -- it only knows
// which pixels came back brighter than white. That is the correct model: a
// flare is light scattering off the elements inside a lens barrel on its way to
// the sensor, so its only input is the image, and its geometry is fixed by the
// lens rather than by the scene.
//
// It also means the effect gates itself. At a low light_intensity nothing in
// the frame clears the bright-pass threshold, the source buffer is black, and
// every term below multiplies out to zero -- no flare, with no separate switch
// to keep in sync. Push the intensity up and the shard's specular starts
// clearing the bar, and the flare arrives on its own.
//
// EDIT ME -- hot-reloaded on save.

in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uSource;   // the bright-pass buffer, at half resolution
uniform vec2 uTexel;         // 1 / that buffer's size
uniform float uStreak;       // streak weight, relative to the ghosts
uniform float uLength;       // streak reach, in source texels per stride unit
uniform float uThreshold;    // second, tighter knee -- see sample() below

// Ghost chain. Spacing is a fraction of the vector to screen centre, so the
// ghosts stay on the line through the centre however the bright spot moves --
// which is the single most recognisable property of a real flare.
const int GHOSTS = 5;
const float GHOST_SPACING = 0.32;
// Radial split between the channels, in texels. Small: a big value reads as a
// chromatic-aberration bug rather than as glass.
// Radial offset between the channels is a *fraction of the ghost's distance
// from centre*, not a fixed texel count. A fixed offset looked right on a
// pinpoint highlight and turned into rainbow banding on a broad one: the three
// channels' rings separate by a constant amount however wide the ring is, so a
// wide ring shows three distinct coloured bands instead of a fringe. Scaling
// with radius keeps the split proportional to what it is splitting.
const float DISPERSAL = 0.012;
const float HALO_WIDTH = 0.42;

// Streak taps. Spaced *geometrically*, not evenly: the decay makes distant taps
// dim anyway, so eight taps at a growing stride reach ~40 texels' worth of
// streak for the cost of eight, where evenly-spaced taps would need forty. The
// slight banding this leaves is taken out by the blur pass that follows.
const int STREAK_TAPS = 8;
const float STREAK_GROWTH = 1.7;
const float STREAK_DECAY = 0.045;

// Weight by distance from the centre, so a ghost fades out as it approaches
// the frame edge instead of being clipped off mid-blob.
float radialFade(vec2 uv, float power) {
    return pow(1.0 - clamp(length(uv - vec2(0.5)) / 0.707, 0.0, 1.0), power);
}

// A second, tighter knee on top of the bright-pass, and the difference between
// a lens flare and a smear.
//
// Bloom and flare want different sources. Glare is what a *bright* pixel does
// to its neighbours, so "brighter than white" is the right bar and a broad lit
// area glowing at its edges is correct. A flare is the image of the aperture,
// reproduced once per lens element -- so its source has to be a small, searing
// *point*. Fed the same buffer as the bloom, a large blown-out facet becomes a
// large blown-out ghost, and the frame turns to soup. Raising the bar here
// keeps only the hottest cores, which are the ones a real lens would ghost.
vec3 sample(vec2 uv) {
    return max(texture(uSource, uv).rgb - uThreshold, vec3(0.0));
}

void main() {
    // Ghosts are sampled from the point mirrored through the centre: a bright
    // spot top-left throws its ghosts bottom-right, as a real lens does.
    vec2 flipped = vec2(1.0) - vUV;
    vec2 ghostVec = (vec2(0.5) - flipped) * GHOST_SPACING;

    vec3 ghosts = vec3(0.0);
    for (int i = 0; i < GHOSTS; i++) {
        vec2 uv = flipped + ghostVec * float(i);
        // Split the channels along the radius. Sampling each channel at its own
        // offset is what gives the ghost a coloured fringe rather than a flat
        // tint, and it costs three taps instead of one.
        vec2 radial = uv - vec2(0.5);
        vec2 spread = radial * DISPERSAL;
        ghosts += vec3(
            sample(uv + spread).r,
            sample(uv).g,
            sample(uv - spread).b
        ) * radialFade(uv, 3.0);
    }
    // Averaged, not summed. Summing made the master strength slider mean
    // something different for every ghost count, and put its useful range in
    // the first tenth of its travel.
    ghosts /= float(GHOSTS);

    // Halo: one ghost pinned at a fixed radius rather than at a fixed spacing,
    // so it stays the same size as the bright spot moves. This is the big soft
    // ring.
    vec2 haloUV = flipped + normalize(ghostVec + 1e-6) * HALO_WIDTH;
    vec3 halo = sample(haloUV) * radialFade(haloUV, 5.0);

    // Streaks: the cross through the bright spot. Two arms at right angles;
    // uStreakAngle is not exposed because a flare's streaks are a property of
    // the aperture blades, which do not rotate.
    vec3 streak = vec3(0.0);
    float weight = 0.0;
    for (int arm = 0; arm < 2; arm++) {
        vec2 axis = arm == 0 ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
        vec2 dir = axis * uTexel * uLength;
        float stride = 1.0;
        for (int i = 0; i < STREAK_TAPS; i++) {
            float decay = exp(-stride * STREAK_DECAY);
            streak += (
                  sample(vUV + dir * stride)
                + sample(vUV - dir * stride)
            ) * decay;
            weight += 2.0 * decay;
            stride *= STREAK_GROWTH;
        }
    }
    streak /= max(weight, 1e-4);

    FragColor = vec4(ghosts + halo * 0.5 + streak * uStreak, 1.0);
}
