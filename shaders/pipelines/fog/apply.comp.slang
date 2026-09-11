// Fog apply pass: samples the froxel volume (baked by pipelines/world/fog.rgen) at each pixel's scene
// depth and blends it over the render-res path-traced image, in place, before DLSS-RR — so
// reconstruction, exposure metering and bloom all see the finished (scene + fog) signal.
//
// The volume stores pre-exposed scene-linear scattering, matching outputImage exactly: scattering is
// additive and transmittance is multiplicative, so the display pass divides one residual exposure out of
// their sum and both stay correct.
import world_common;
import bindings;
import fog;

[[vk::push_constant]] PushAddr pcAddr;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    WorldPush push = ConstPtr<WorldPush>(pcAddr.worldPushAddr)[0];
    int2 pix = int2(dispatchId.xy);
    uint w;
    uint h;
    outputImage.GetDimensions(w, h);
    if (pix.x >= int(w) || pix.y >= int(h)) {
        return;
    }
    float maxDist = max(push.fogParams0.z, FOG_NEAR + 1.0);
    // Reversed-Z hardware depth (near = 1, far = 0) -> view depth through the projection's third row:
    // depth = (A * viewZ + B) / -viewZ, so -viewZ = B / (depth + A).
    float projA = push.curViewProj[2][2];
    float projB = push.curViewProj[2][3];
    float viewDepth = projB / max(depthImage[pix] + projA, 1.0e-6);
    float slice = clamp(fogViewDepthToSlice(viewDepth, 1.0, maxDist), 0.0, 1.0);
    float2 uv = (float2(pix) + 0.5) / float2(float(w), float(h));
    float4 fogSample = fogVolume.SampleLevel(float3(uv, slice), 0.0);
    float4 scene = outputImage[pix];
    outputImage[pix] = float4(scene.rgb * fogSample.a + fogSample.rgb, scene.a);
}
