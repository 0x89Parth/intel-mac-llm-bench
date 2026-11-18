#include <metal_stdlib>
using namespace metal;

// Row-major FP16 weights, a single FP32 activation vector and FP32 outputs.
kernel void projection_row(device const half *weights [[buffer(0)]],
                           device const float *input [[buffer(1)]],
                           device float *output [[buffer(2)]],
                           constant uint2 &shape [[buffer(3)]],
                           uint row [[thread_position_in_grid]]) {
    if (row >= shape.x) return;
    uint base = row * shape.y;
    float sum = 0;
    for (uint col = 0; col < shape.y; ++col) {
        sum = fma(float(weights[base + col]), input[col], sum);
    }
    output[row] = sum;
}

// One SIMD group cooperates on each row. Adjacent lanes read adjacent weights.
kernel void projection_simd(device const half *weights [[buffer(0)]],
                            device const float *input [[buffer(1)]],
                            device float *output [[buffer(2)]],
                            constant uint2 &shape [[buffer(3)]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint groupWithinBlock [[simdgroup_index_in_threadgroup]],
                            uint groupsInBlock [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint width [[threads_per_simdgroup]]) {
    uint row = group.x * groupsInBlock + groupWithinBlock;
    if (row >= shape.x) return;
    uint base = row * shape.y;
    float partial = 0;
    for (uint col = lane; col < shape.y; col += width) {
        partial = fma(float(weights[base + col]), input[col], partial);
    }
    float sum = simd_sum(partial);
    if (lane == 0) output[row] = sum;
}
