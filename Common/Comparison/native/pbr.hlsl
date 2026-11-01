// Linear-light Cook-Torrance PBR, hardware rasterization and shadow-map PCF.
struct Light { row_major float4x4 vp; float4 positionPower; float4 color; };
cbuffer Camera : register(b0) {
    row_major float4x4 viewProjection;
    float4 eye;
    float mipBias;
    uint normalMode;
    uint pointCount;
    float padding;
};
cbuffer ShadowPass : register(b1) { uint shadowIndex; };
Texture2D<float4> Diffuse : register(t0);
Texture2D<float4> Normal : register(t1);
Texture2D<float4> Roughness : register(t2);
Texture2D<float4> Occlusion : register(t3);
Texture2D<float4> Metallic : register(t4);
Texture2D<float4> Specular : register(t5);
Texture2DArray<float> Shadows : register(t6);
StructuredBuffer<Light> Lights : register(t7);
SamplerState MaterialSampler : register(s0);
SamplerComparisonState ShadowSampler : register(s1);

struct Vertex {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float4 tangent : TANGENT;
    float2 uv : TEXCOORD;
};
struct Interpolants {
    float4 clip : SV_Position;
    float3 position : POSITION;
    float3 normal : NORMAL;
    float4 tangent : TANGENT;
    float2 uv : TEXCOORD;
};
Interpolants VS(Vertex v) {
    Interpolants o;
    o.clip = mul(float4(v.position, 1), viewProjection);
    o.position = v.position; o.normal = v.normal; o.tangent = v.tangent; o.uv = v.uv;
    return o;
}
float4 ShadowVS(Vertex v) : SV_Position { return mul(float4(v.position, 1), Lights[shadowIndex].vp); }

float3 unit(float3 v) { return v * rsqrt(max(dot(v, v), 1e-12)); }

float3 decodeNormal(float3 encoded) {
    float3 n = encoded * 2 - 1;
    if (normalMode == 1) {
        float2 xy = n.xy * sqrt(min(1., .9999 / max(dot(n.xy, n.xy), 1e-8)));
        n = float3(xy, sqrt(max(1 - dot(xy, xy), .0001)));
    } else if (normalMode == 2) {
        float2 xy = float2(n.x + n.y, n.x - n.y) * .5;
        n = float3(xy, 1 - abs(xy.x) - abs(xy.y));
    }
    return unit(n);
}

float visibility(float3 position, float3 geometricNormal, uint index) {
    // Offset with the geometric normal; normal-map detail must not move geometry.
    float4 clip = mul(float4(position + geometricNormal * .0015, 1), Lights[index].vp);
    float3 p = clip.xyz / clip.w;
    float2 uv = p.xy * float2(.5, -.5) + .5;
    if (clip.w <= 0 || any(uv < 0) || any(uv > 1) || p.z < 0 || p.z > 1) return 1;
    uint width, height, layers;
    Shadows.GetDimensions(width, height, layers);
    float2 texel = 1. / float2(width, height);
    float sum = 0;
    [unroll] for (int y = -1; y <= 1; y++)
        [unroll] for (int x = -1; x <= 1; x++)
            sum += Shadows.SampleCmpLevelZero(ShadowSampler, float3(uv + float2(x, y) * texel, index), p.z - .00005);
    return sum / 9.;
}
uint cubeFace(float3 d) {
    float3 a = abs(d);
    if (a.x >= a.y && a.x >= a.z) return d.x >= 0 ? 0 : 1;
    if (a.y >= a.z) return d.y >= 0 ? 2 : 3;
    return d.z >= 0 ? 4 : 5;
}
float3 brdf(float3 albedo, float rough, float metal, float3 f0, float3 n, float3 view, float3 light) {
    float nl = saturate(dot(n, light)), nv = max(saturate(dot(n, view)), 1e-4);
    float3 halfVector = unit(view + light);
    float nh = saturate(dot(n, halfVector)), vh = saturate(dot(view, halfVector));
    float alpha2 = pow(rough, 4);
    float denominator = nh * nh * (alpha2 - 1) + 1;
    float distribution = alpha2 / max(3.14159265 * denominator * denominator, 1e-10);
    float k = (rough + 1) * (rough + 1) / 8;
    float geometry = nv / (nv * (1 - k) + k) * nl / max(nl * (1 - k) + k, 1e-6);
    float3 fresnel = f0 + (1 - f0) * pow(1 - vh, 5);
    return ((1 - fresnel) * (1 - metal) * albedo / 3.14159265 +
            distribution * geometry * fresnel / max(4 * nv * nl, 1e-5)) * nl;
}
float3 linearToSRGB(float3 x) {
    return float3(x.r <= .0031308 ? 12.92*x.r : 1.055*pow(x.r,1/2.4)-.055,
                  x.g <= .0031308 ? 12.92*x.g : 1.055*pow(x.g,1/2.4)-.055,
                  x.b <= .0031308 ? 12.92*x.b : 1.055*pow(x.b,1/2.4)-.055);
}
float4 PS(Interpolants i) : SV_Target {
    float3 albedo = saturate(Diffuse.SampleBias(MaterialSampler, i.uv, mipBias).rgb);
    float rough = clamp(Roughness.SampleBias(MaterialSampler, i.uv, mipBias).r, .045, 1);
    float ao = saturate(Occlusion.SampleBias(MaterialSampler, i.uv, mipBias).r);
    float metal = saturate(Metallic.SampleBias(MaterialSampler, i.uv, mipBias).r);
    float spec = saturate(Specular.SampleBias(MaterialSampler, i.uv, mipBias).r);
    float3 gn = unit(i.normal);
    float3 t = unit(i.tangent.xyz - gn * dot(gn, i.tangent.xyz));
    float3 b = cross(gn, t) * i.tangent.w;
    float3 nts = decodeNormal(Normal.SampleBias(MaterialSampler, i.uv, mipBias).rgb);
    // Source normals are OpenGL (+Y up); texture image V increases downward.
    float3 n = unit(t * nts.x - b * nts.y + gn * nts.z);
    float3 view = unit(eye.xyz - i.position);
    float3 f0 = lerp((.02 + .14 * spec).xxx, albedo, metal);
    // AO attenuates the small ambient fill, never the direct light.
    float3 color = albedo * .12 * ao * (1 - metal);
    if (Lights[0].positionPower.w > 0) {
        float3 l = unit(Lights[0].positionPower.xyz);
        color += brdf(albedo, rough, metal, f0, n, view, l) * Lights[0].color.rgb *
                 Lights[0].positionPower.w * visibility(i.position, gn, 0);
    }
    [loop] for (uint index = 0; index < pointCount; index++) {
        uint base = 1 + index * 6;
        float3 delta = Lights[base].positionPower.xyz - i.position;
        float3 l = unit(delta);
        float power = Lights[base].positionPower.w / max(dot(delta, delta), .04);
        color += brdf(albedo, rough, metal, f0, n, view, l) * Lights[base].color.rgb * power *
                 visibility(i.position, gn, base + cubeFace(-delta));
    }
    color = color / (1 + color);
    return float4(saturate(linearToSRGB(color)), 1);
}
