#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

static void require(BOOL condition, NSString *message) {
    if (!condition) { fprintf(stderr, "%s\n", message.UTF8String); exit(1); }
}

static uint32_t randomBits(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *state = x;
}

static id<MTLBuffer> buffer(id<MTLDevice> device, NSUInteger length, MTLResourceOptions mode) {
    id<MTLBuffer> result = [device newBufferWithLength:length options:mode];
    require(result != nil, @"Metal buffer allocation failed");
    return result;
}

static void finish(id<MTLCommandBuffer> cmd) {
    [cmd commit];
    [cmd waitUntilCompleted];
    require(cmd.status == MTLCommandBufferStatusCompleted, cmd.error.description ?: @"GPU command failed");
}

static NSDictionary *runKernel(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                               id<MTLBuffer> weights, id<MTLBuffer> input, id<MTLBuffer> output,
                               vector_uint2 shape, BOOL cooperative) {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:weights offset:0 atIndex:0];
    [enc setBuffer:input offset:0 atIndex:1];
    [enc setBuffer:output offset:0 atIndex:2];
    [enc setBytes:&shape length:sizeof(shape) atIndex:3];
    NSUInteger width = pipeline.threadExecutionWidth;
    NSUInteger groupsPerBlock = MIN(4, pipeline.maxTotalThreadsPerThreadgroup / width);
    NSUInteger threads = width * groupsPerBlock;
    NSUInteger rowsPerBlock = cooperative ? groupsPerBlock : threads;
    NSUInteger blocks = (shape.x + rowsPerBlock - 1) / rowsPerBlock;
    [enc dispatchThreadgroups:MTLSizeMake(blocks,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    [enc endEncoding];
    uint64_t start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    finish(cmd);
    double wallMs = (clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start) / 1e6;
    double gpuMs = (cmd.GPUEndTime - cmd.GPUStartTime) * 1e3;
    return @{@"gpu_ms": @(gpuMs), @"wall_ms": @(wallMs)};
}

static NSDictionary *check(id<MTLBuffer> result, const double *ref, NSUInteger rows) {
    const float *actual = result.contents;
    double error2 = 0, reference2 = 0, maxAbs = 0, maxRel = 0;
    NSUInteger failures = 0;
    for (NSUInteger i = 0; i < rows; ++i) {
        if (!isfinite(actual[i])) { ++failures; continue; }
        double delta = fabs(actual[i] - ref[i]);
        maxAbs = fmax(maxAbs, delta);
        if (fabs(ref[i]) >= 0.1) maxRel = fmax(maxRel, delta/fabs(ref[i]));
        failures += delta > 5e-4 + 2e-4*fabs(ref[i]);
        error2 += delta * delta;
        reference2 += ref[i] * ref[i];
    }
    return @{@"max_abs_error": @(maxAbs), @"relative_l2_error": @(sqrt(error2/reference2)),
             @"max_relative_error_where_abs_ref_ge_0.1": @(maxRel), @"failed_rows": @(failures)};
}

static double median(NSArray<NSNumber *> *values) {
    NSArray *sorted = [values sortedArrayUsingSelector:@selector(compare:)];
    NSUInteger count = sorted.count;
    return ([sorted[(count-1)/2] doubleValue] + [sorted[count/2] doubleValue]) / 2;
}

static NSDictionary *runShape(id<MTLDevice> device, id<MTLCommandQueue> queue,
                               NSArray<id<MTLComputePipelineState>> *pipelines,
                               vector_uint2 shape, NSUInteger trials) {
    const NSUInteger n = shape.x, k = shape.y;
    id<MTLBuffer> weights = buffer(device, n*k*sizeof(uint16_t), MTLResourceStorageModeManaged);
    id<MTLBuffer> input = buffer(device, k*sizeof(float), MTLResourceStorageModeManaged);
    uint16_t *bits = weights.contents;
    float *x = input.contents;
    NSMutableData *reference = [NSMutableData dataWithLength:n*sizeof(double)];
    double *ref = reference.mutableBytes;
    uint32_t seed = 0x1ae774bc;
    for (NSUInteger col = 0; col < k; ++col) {
        x[col] = ((int32_t)(randomBits(&seed) >> 8) - 8388608) / 8388608.f;
    }
    for (NSUInteger row = 0; row < n; ++row) {
        double sum = 0;
        for (NSUInteger col = 0; col < k; ++col) {
            uint32_t word = randomBits(&seed);
            unsigned exponent = 11 + ((word >> 10) & 3);
            unsigned mantissa = word & 1023;
            bits[row*k+col] = (uint16_t)((word & 0x8000) | (exponent << 10) | mantissa);
            double magnitude = ldexp((double)(1024 + mantissa), (int)exponent-25);
            double weight = (word & 0x8000) ? -magnitude : magnitude;
            sum += weight * x[col];
        }
        ref[row] = sum;
    }
    [weights didModifyRange:NSMakeRange(0,weights.length)];
    [input didModifyRange:NSMakeRange(0,input.length)];
    NSArray *outputs = @[buffer(device,n*sizeof(float),MTLResourceStorageModeManaged),
                         buffer(device,n*sizeof(float),MTLResourceStorageModeManaged)];
    for (id<MTLBuffer> out in outputs) {
        float *v = out.contents; for (NSUInteger i=0; i<n; ++i) v[i] = NAN;
        [out didModifyRange:NSMakeRange(0,out.length)];
    }
    NSMutableArray *samples[2] = {[NSMutableArray array], [NSMutableArray array]};
    for (NSUInteger pair=0; pair<3+trials; ++pair) {
        for (NSUInteger turn=0; turn<2; ++turn) { @autoreleasepool {
            NSUInteger j = turn ^ (pair & 1);
            NSDictionary *sample = runKernel(queue,pipelines[j],weights,input,outputs[j],shape,j==1);
            if (pair>=3) [samples[j] addObject:sample];
        }}
    }
    id<MTLCommandBuffer> sync = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [sync blitCommandEncoder];
    for (id<MTLBuffer> out in outputs) [blit synchronizeResource:out];
    [blit endEncoding];
    finish(sync);
    NSMutableArray *results = [NSMutableArray array];
    BOOL gpuTimes = YES;
    for (NSUInteger j=0; j<2; ++j) {
        for (NSDictionary *sample in samples[j]) if ([sample[@"gpu_ms"] doubleValue]<=0) gpuTimes = NO;
    }
    NSString *metric = gpuTimes ? @"gpu_ms" : @"wall_ms";
    for (NSUInteger j=0; j<2; ++j) {
        NSDictionary *accuracy = check(outputs[j],ref,n);
        [results addObject:@{@"kernel":j==0 ? @"projection_row" : @"projection_simd",
            @"median_ms": @(median([samples[j] valueForKey:metric])), @"samples":samples[j],
            @"thread_execution_width":@(pipelines[j].threadExecutionWidth), @"accuracy":accuracy}];
        require([accuracy[@"failed_rows"] integerValue]==0,
            [NSString stringWithFormat:@"Numerical verification failed at %lux%lu for %@: %@",n,k,results.lastObject[@"kernel"],accuracy]);
    }
    return @{@"rows":@(n),@"columns":@(k),@"clock":metric,@"kernels":results,
             @"speedup_from_medians":@([results[0][@"median_ms"] doubleValue]/[results[1][@"median_ms"] doubleValue])};
}

int main(int argc, char **argv) { @autoreleasepool {
    require(argc==2, @"Usage: projection source.metal");
    NSError *err = nil;
    id<MTLDevice> device = nil;
    for(id<MTLDevice> candidate in MTLCopyAllDevices()) {
        if ([candidate.name containsString:@"Radeon Pro 560X"]) {device = candidate; break;}
    }
    require(device!=nil,@"Required AMD Radeon Pro 560X was not found.");
    NSString *source = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&err];
    require(source!=nil,err.description);
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.fastMathEnabled = NO;
    options.languageVersion = MTLLanguageVersion2_2;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&err];
    require(library!=nil,err.description);
    NSMutableArray *pipelines = [NSMutableArray array];
    for(NSString *name in @[@"projection_row",@"projection_simd"]) {
        id<MTLFunction> function = [library newFunctionWithName:name];
        require(function!=nil,@"Missing Metal function");
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&err];
        require(pipeline!=nil,err.description);
        [pipelines addObject:pipeline];
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSMutableArray *runs = [NSMutableArray array];
    vector_uint2 shapes[] = {{2048,2048},{2051,2053},{4096,4096}};
    for (NSUInteger i=0; i<sizeof(shapes)/sizeof(shapes[0]); ++i) { @autoreleasepool {
        [runs addObject:runShape(device,queue,pipelines,shapes[i],12)];
    }}
    NSDictionary *report=@{@"device":device.name,@"device_registry_id":@(device.registryID),
        @"os":NSProcessInfo.processInfo.operatingSystemVersionString,@"weights":@"binary16", @"activations":@"float32",
        @"reference":@"CPU double accumulator", @"runtime_compiler":@"MTLDevice newLibraryWithSource, MSL 2.2, fastMathEnabled=NO",
        @"warmup_pairs":@3,@"measured_pairs":@12,@"order":@"alternating AB and BA pairs",
        @"seed":@"0x1ae774bc",@"results":runs};
    NSData *json=[NSJSONSerialization dataWithJSONObject:report options:(NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys) error:&err];
    require(json!=nil,err.description);
    fwrite(json.bytes,1,json.length,stdout); putchar('\n');
    return 0;
}}
