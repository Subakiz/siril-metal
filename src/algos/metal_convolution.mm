/*
 * This file is part of Siril, an astronomy image processor.
 * Copyright (C) 2005-2011 Francois Meyer (dulle at free.fr)
 * Copyright (C) 2012-2026 team free-astro (see more in AUTHORS file)
 * Reference site is https://siril.org
 *
 * Siril is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Siril is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Siril. If not, see <http://www.gnu.org/licenses/>.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <climits>
#include <cstdint>
#include <cstring>

#include "metal_convolution.h"

namespace {
const char *shader_source = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void convolve_float(device const float *input [[buffer(0)]],
                          device const float *weights [[buffer(1)]],
                          device float *output [[buffer(2)]],
                          constant uint *p [[buffer(3)]],
                          uint index [[thread_position_in_grid]]) {
    if (index >= p[3]) return;
    const int width = int(p[0]), height = int(p[1]), size = int(p[2]);
    const uint plane_size = p[0] * p[1];
    const uint plane = index / plane_size;
    const int x = int(index % p[0]);
    const int y = int((index % plane_size) / p[0]);
    const int radius = size / 2;
    float sum = 0.0f;
    // Preserve the CPU's correlation orientation and accumulation order.
    for (int m = -radius; m <= radius; ++m) {
        const int sx = x + m;
        if (sx < 0 || sx >= width) continue;
        for (int n = -radius; n <= radius; ++n) {
            const int sy = y + n;
            if (sy >= 0 && sy < height)
                sum += input[plane * plane_size + uint(sy * width + sx)] *
                       weights[(n + radius) * size + m + radius];
        }
    }
    output[index] = sum;
}
)metal";

struct Backend {
	id<MTLDevice> device = nil;
	id<MTLCommandQueue> queue = nil;
	id<MTLComputePipelineState> pipeline = nil;

	Backend() {
		device = MTLCreateSystemDefaultDevice();
		if (!device) return;
		MTLCompileOptions *options = [[[MTLCompileOptions alloc] init] autorelease];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
		if (@available(macOS 15.0, *))
			options.mathMode = MTLMathModeSafe;
		else
#endif
		{
			// Fast math off keeps results bit-compatible with the CPU path.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
			options.fastMathEnabled = NO;
#pragma clang diagnostic pop
		}
		id<MTLLibrary> library = [[device newLibraryWithSource:
				[NSString stringWithUTF8String:shader_source]
				options:options error:nil] autorelease];
		if (!library) return;
		id<MTLFunction> function = [[library newFunctionWithName:@"convolve_float"] autorelease];
		if (!function) return;
		pipeline = [device newComputePipelineStateWithFunction:function error:nil];
		if (pipeline) queue = [device newCommandQueue];
	}

	~Backend() {
		[queue release];
		[pipeline release];
		[device release];
	}
};
}

int siril_metal_convolve_float(const float *input, const float *kernel,
		float *output, int width, int height, int channels, int kernel_size) {
	// Kernels above 31 stay on CPU until larger PSFs are benchmarked.
	if (!input || !kernel || !output || width <= 0 || height <= 0 || channels <= 0 ||
			kernel_size < 3 || kernel_size > 31 || !(kernel_size & 1) ||
			kernel_size > width || kernel_size > height)
		return 0;
	const uint64_t plane_size = uint64_t(width) * uint64_t(height);
	if (plane_size > INT_MAX || uint64_t(channels) > INT_MAX / plane_size)
		return 0;
	const uint32_t count = uint32_t(plane_size * channels);
	// Small convolutions cost less on CPU than Metal allocation and submission.
	if (kernel_size < 7 || uint64_t(count) * kernel_size * kernel_size < 32ULL * 1024 * 1024)
		return 0;
	const size_t bytes = size_t(count) * sizeof(float);
	const size_t kernel_bytes = size_t(kernel_size) * kernel_size * sizeof(float);

	@autoreleasepool {
		static Backend backend;
		if (!backend.queue || bytes > backend.device.maxBufferLength)
			return 0;
		id<MTLBuffer> source = [[backend.device newBufferWithBytes:input
				length:bytes options:MTLResourceStorageModeShared] autorelease];
		id<MTLBuffer> weights = [[backend.device newBufferWithBytes:kernel
				length:kernel_bytes options:MTLResourceStorageModeShared] autorelease];
		id<MTLBuffer> destination = [[backend.device newBufferWithLength:bytes
				options:MTLResourceStorageModeShared] autorelease];
		if (!source || !weights || !destination) return 0;
		id<MTLCommandBuffer> command = [backend.queue commandBuffer];
		id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
		if (!command || !encoder) return 0;
		const uint32_t parameters[] = { uint32_t(width), uint32_t(height),
				uint32_t(kernel_size), count };
		[encoder setComputePipelineState:backend.pipeline];
		[encoder setBuffer:source offset:0 atIndex:0];
		[encoder setBuffer:weights offset:0 atIndex:1];
		[encoder setBuffer:destination offset:0 atIndex:2];
		[encoder setBytes:parameters length:sizeof(parameters) atIndex:3];
		const NSUInteger group_size = backend.pipeline.threadExecutionWidth;
		[encoder dispatchThreadgroups:MTLSizeMake((count + group_size - 1) / group_size, 1, 1)
				threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
		[encoder endEncoding];
		[command commit];
		[command waitUntilCompleted];
		if (command.status != MTLCommandBufferStatusCompleted) return 0;
		std::memcpy(output, destination.contents, bytes);
		return 1;
	}
}
