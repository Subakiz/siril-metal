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
#ifndef SIRIL_METAL_CONVOLUTION_H
#define SIRIL_METAL_CONVOLUTION_H

#ifdef __cplusplus
extern "C" {
#endif

/* Planar float correlation with zero borders, matching img_t::conv2.
 * Returns 1 on success, or 0 to use the CPU with output untouched.
 * Input and output may alias. */
int siril_metal_convolve_float(const float *input, const float *kernel,
		float *output, int width, int height, int channels, int kernel_size);

#ifdef __cplusplus
}
#endif
#endif
