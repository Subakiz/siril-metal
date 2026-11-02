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

/*
 * Synthetic correctness checks and an optional end-to-end conv2 benchmark.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "algos/img_t/image.hpp"
#ifdef HAVE_METAL
#include "algos/metal_convolution.h"
#import <Metal/Metal.h>
#endif

cominfo com;
int sequence_is_running = 0;

static void require(bool condition, const char *message) {
	if (!condition) {
		std::fprintf(stderr, "FAIL: %s\n", message);
		std::exit(1);
	}
}

static img_t<float> image(int width, int height, int channels, int seed) {
	img_t<float> result(width, height, channels);
	for (int c = 0; c < channels; ++c)
		for (int y = 0; y < height; ++y)
			for (int x = 0; x < width; ++x)
				result(x, y, c) = ((x * 17 + y * 31 + c * 43 + seed * 13) % 251 - 125) / 127.f;
	return result;
}

static img_t<float> kernel(int size, int seed) {
	img_t<float> result(size, size, 1);
	float sum = 0.f;
	for (int y = 0; y < size; ++y)
		for (int x = 0; x < size; ++x) {
			const float value = 1.f + (x * 3 + y * 7 + seed * 11) % 19;
			result(x, y) = value;
			sum += value;
		}
	for (float &value : result.data)
		value /= sum;
	return result;
}

// Siril's conv2 is correlation (no kernel flip), with zero padding.
static img_t<float> reference(const img_t<float>& input, const img_t<float>& weights) {
	img_t<float> result(input.w, input.h, input.d);
	const int radius = weights.w / 2;
	for (int c = 0; c < input.d; ++c)
		for (int y = 0; y < input.h; ++y)
			for (int x = 0; x < input.w; ++x) {
				float sum = 0.f;
				for (int dx = -radius; dx <= radius; ++dx)
					for (int dy = -radius; dy <= radius; ++dy)
						if (x + dx >= 0 && x + dx < input.w && y + dy >= 0 && y + dy < input.h)
							sum += input(x + dx, y + dy, c) * weights(dx + radius, dy + radius);
				result(x, y, c) = sum;
			}
	return result;
}

static void compare(const img_t<float>& actual, const img_t<float>& expected, const char *label) {
	require(actual.similar(expected), "output shape");
	float maximum_error = 0.f;
	for (size_t i = 0; i < actual.data.size(); ++i) {
		const float error = std::abs(actual.data[i] - expected.data[i]);
		if (!std::isfinite(actual.data[i]) || error > 2e-5f + 2e-5f * std::abs(expected.data[i])) {
			std::fprintf(stderr, "FAIL: %s pixel %zu: %.9g expected %.9g\n", label, i, actual.data[i], expected.data[i]);
			std::exit(1);
		}
		maximum_error = std::max(maximum_error, error);
	}
	std::printf("PASS: %s (%dx%dx%d, max absolute error %.3g)\n", label, actual.w, actual.h, actual.d, maximum_error);
}

#ifdef HAVE_METAL
static int metal(const img_t<float>& input, const img_t<float>& weights, img_t<float>& output) {
	return siril_metal_convolve_float(input.data.data(), weights.data.data(), output.data.data(),
			input.w, input.h, input.d, weights.w);
}

static void concurrent_calls() {
	std::array<std::thread, 4> workers;
	for (int n = 0; n < 4; ++n)
		workers[n] = std::thread([n] {
			auto input = image(521, 509, 1, n + 20);
			auto weights = kernel(13, n + 10);
			img_t<float> output(input.w, input.h, input.d);
			require(metal(input, weights, output) == 1, "concurrent Metal call must succeed");
			compare(output, reference(input, weights), "concurrent Metal");
		});
	for (auto& worker : workers)
		worker.join();
}

static void invalid_arguments() {
	float input[9] = {}, weights[9] = {}, output[9];
	std::fill(output, output + 9, 123.f);
	auto rejected = [&](const float *in, const float *k, float *out, int w, int h, int d, int ks) {
		require(siril_metal_convolve_float(in, k, out, w, h, d, ks) == 0, "invalid arguments must fall back");
		for (float value : output)
			require(value == 123.f, "fallback must leave output untouched");
	};
	rejected(nullptr, weights, output, 3, 3, 1, 3);
	rejected(input, nullptr, output, 3, 3, 1, 3);
	rejected(input, weights, nullptr, 3, 3, 1, 3);
	rejected(input, weights, output, 0, 3, 1, 3);
	rejected(input, weights, output, 3, -1, 1, 3);
	rejected(input, weights, output, 3, 3, 0, 3);
	rejected(input, weights, output, 3, 3, 1, 2);
	rejected(input, weights, output, 3, 3, 1, 1);
	rejected(input, weights, output, 3, 3, 1, 5);
	rejected(input, weights, output, 33, 33, 1, 33);
	rejected(input, weights, output, INT_MAX, 3, 1, 3);
	rejected(input, weights, output, INT_MAX, INT_MAX, 1, 3);
	rejected(input, weights, output, 3, 3, INT_MAX, 3);
	rejected(input, weights, output, 3, 3, 1, 3); // Valid small workload falls back too.
	std::puts("PASS: invalid arguments preserve output");
}
#endif

static void non_float_fallback() {
	auto input = image(17, 19, 1, 9);
	auto weights = kernel(3, 7);
	auto expected = reference(input, weights);
	img_t<double> input64(input.w, input.h, input.d), weights64(3, 3, 1), output64(input.w, input.h, input.d);
	std::copy(input.data.begin(), input.data.end(), input64.data.begin());
	std::copy(weights.data.begin(), weights.data.end(), weights64.data.begin());
	output64.conv2(input64, weights64);
	img_t<float> mixed(input.w, input.h, input.d);
	mixed.conv2(input, weights64);
	compare(mixed, expected, "mixed kernel type CPU fallback");
	for (size_t i = 0; i < expected.data.size(); ++i)
		require(std::abs(output64.data[i] - expected.data[i]) < 2e-5, "double precision CPU fallback");
	std::puts("PASS: double precision CPU fallback");
}

static void benchmark() {
	for (int side : {64, 128, 256, 512, 1024}) {
	for (int ks : {3, 5, 7, 13}) {
		auto input = image(side, side, 1, 7);
		auto weights = kernel(ks, 4);
		img_t<float> output(input.w, input.h, input.d);
		const auto cold_start = std::chrono::steady_clock::now();
		output.conv2(input, weights);
		const auto warm_start = std::chrono::steady_clock::now();
		constexpr int repeats = 30;
		std::vector<double> samples;
		for (int n = 0; n < repeats; ++n) {
			const auto start = std::chrono::steady_clock::now();
			output.conv2(input, weights);
			samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
		}
		std::sort(samples.begin(), samples.end());
		const auto end = std::chrono::steady_clock::now();
		const double cold_ms = std::chrono::duration<double, std::milli>(warm_start - cold_start).count();
		const double warm_ms = std::chrono::duration<double, std::milli>(end - warm_start).count() / repeats;
#ifdef HAVE_METAL
		const char *mode = "Metal-enabled";
#else
		const char *mode = "CPU";
#endif
		std::printf("BENCH: %s actual img_t::conv2 %dx%d mono k%d cold %.3f ms warm mean %.3f ms median %.3f ms (allocation/transfers included), sample %.9g\n",
				mode, side, side, ks, cold_ms, warm_ms, samples[repeats / 2], output.data[output.size / 2]);
	}
	}
}

int main(int argc, char **argv) {
	com.max_thread = std::max(2u, std::thread::hardware_concurrency());
#ifdef _OPENMP
	std::printf("OpenMP enabled, configured CPU threads: %d\n", com.max_thread);
#else
	std::puts("OpenMP disabled");
#endif
	if (argc == 2 && std::strcmp(argv[1], "--benchmark") == 0) {
		benchmark();
		return 0;
	}
	require(argc == 1, "usage: metal_convolution_test [--benchmark]");
	non_float_fallback();
#ifdef HAVE_METAL
	invalid_arguments();
	// Large enough to exercise the backend instead of its small-workload fallback.
	auto probe = image(1024, 1024, 1, 0);
	auto probe_kernel = kernel(7, 0);
	img_t<float> probe_output(probe.w, probe.h, probe.d);
	id<MTLDevice> device = MTLCreateSystemDefaultDevice();
	if (!device) {
		std::puts("SKIP: no Metal device");
		return 77;
	}
	std::printf("Metal device: %s\n", [[device name] UTF8String]);
	[device release];
	require(metal(probe, probe_kernel, probe_output) == 1, "Metal backend failed on an available device");
	compare(probe_output, reference(probe, probe_kernel), "Metal mono probe");
	concurrent_calls();
#endif
	for (int width : {17, 521}) {
	for (int channels : {1, 3}) {
		for (int ks : {3, 5, 7, 13}) {
			for (int seed : {1, 2}) {
				auto input = image(width, width == 17 ? 19 : 509, channels, seed);
				auto weights = kernel(ks, seed);
				auto expected = reference(input, weights);
				img_t<float> output(input.w, input.h, input.d);
				output.conv2(input, weights);
				compare(output, expected, "img_t::conv2");
				auto alias = input;
				alias.conv2(alias, weights);
				compare(alias, expected, "img_t::conv2 in-place");
#ifdef HAVE_METAL
				if (metal(input, weights, output)) {
					compare(output, expected, "direct Metal");
					alias = input;
					require(metal(alias, weights, alias) == 1, "Metal in-place workload must succeed");
					compare(alias, expected, "direct Metal in-place");
				}
#endif
			}
		}
	}
	}
	return 0;
}
