#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <chrono>
#include <omp.h>
#include <immintrin.h>

void sepiaMP(unsigned char* data, int width, int height, int channels) {
#pragma omp parallel
    {
        int x, red, green, blue;
#pragma omp for
        for (int i = 0; i < width * height; i++) {
            x = i * channels;
            red = data[x];
            green = data[x + 1];
            blue = data[x + 2];
            data[x] = ((red * 0.393) + (green * 0.769) + (blue * 0.189) > 255) ? 255 : (red * 0.393) + (green * 0.769) + (blue * 0.189);
            data[x + 1] = ((red * 0.349) + (green * 0.686) + (blue * 0.168) > 255) ? 255 : (red * 0.349) + (green * 0.686) + (blue * 0.168);
            data[x + 2] = ((red * 0.272) + (green * 0.534) + (blue * 0.131) > 255) ? 255 : (red * 0.272) + (green * 0.534) + (blue * 0.131);
        }
    }
}

void sepia(unsigned char* data, int width, int height, int channels) {
    int x, red, green, blue;
    for (int i = 0; i < width * height; i++) {
        x = i * channels;
        red = data[x];
        green = data[x + 1];
        blue = data[x + 2];
        data[x] = ((red * 0.393) + (green * 0.769) + (blue * 0.189) > 255) ? 255 : (red * 0.393) + (green * 0.769) + (blue * 0.189);
        data[x + 1] = ((red * 0.349) + (green * 0.686) + (blue * 0.168) > 255) ? 255 : (red * 0.349) + (green * 0.686) + (blue * 0.168);
        data[x + 2] = ((red * 0.272) + (green * 0.534) + (blue * 0.131) > 255) ? 255 : (red * 0.272) + (green * 0.534) + (blue * 0.131);
    }
}

void posterization(unsigned char* data, int width, int height, int channels) {
    int x;
    for (int i = 0; i < width * height; i++) {
        x = i * channels;
        for (int k = 0; k < 3; k++) {
            data[x + k] = (data[x + k] / 64) * 64;
        }
    }
}

void posterizationMP(unsigned char* data, int width, int height, int channels) {
#pragma omp parallel
    {
        int x;
#pragma omp for
        for (int i = 0; i < width * height; i++) {
            x = i * channels;
            for (int k = 0; k < 3; k++) {
                data[x + k] = (data[x + k] / 64) * 64;
            }
        }
    }
}

void solarization(unsigned char* data, int width, int height, int channels) {
    int x;
    for (int i = 0; i < height * width; i++) {
        x = i * channels;
        for (int k = 0; k < 3; k++) {
            if (data[x + k] > 128) {
                data[x + k] = 255 - data[x + k];
            }
        }
    }
}

void solarizationMP(unsigned char* data, int width, int height, int channels) {
#pragma omp parallel
    {
        int x;
#pragma omp for
        for (int i = 0; i < height * width; i++) {
            x = i * channels;
            for (int k = 0; k < 3; k++) {
                if (data[x + k] > 128) {
                    data[x + k] = 255 - data[x + k];
                }
            }
        }
    }
}

void posterizationVector(unsigned char* data, int width, int height, int channels) {
    int i = 0;
    __m256i mask;
    if (channels == 4) {
        mask = _mm256_set_epi8(
            0, -1, -1, -1,
            0, -1, -1, -1,
            0, -1, -1, -1,
            0, -1, -1, -1,
            0, -1, -1, -1,
            0, -1, -1, -1,
            0, -1, -1, -1,
            0, -1, -1, -1
        );
    }
    for (; i + 32 <= width * height * channels; i = i + 32) {
        __m256i pixels = _mm256_loadu_si256((__m256i*) & data[i]);
        __m256i maskForPost = _mm256_set1_epi8(0xC0);
        __m256i posterized = _mm256_and_si256(pixels, maskForPost);
        __m256i result = posterized;
        if (channels == 4) {
            result = _mm256_blendv_epi8(pixels, posterized, mask);
        }
        _mm256_storeu_si256((__m256i*) & data[i], result);
    }
    for (; i < width * height * channels; ++i) {
        data[i] = data[i] & 0xC0;
    }
}


void solarizationVector(unsigned char* data, int width, int height, int channels) {
    int i = 0;
    for (; i + 32 <= width * height * channels; i += 32) {
        __m256i mask;
        if (channels == 4) {
            mask = _mm256_set_epi8(
                0, -1, -1, -1,
                0, -1, -1, -1,
                0, -1, -1, -1,
                0, -1, -1, -1,
                0, -1, -1, -1,
                0, -1, -1, -1,
                0, -1, -1, -1,
                0, -1, -1, -1
            );
        }

        __m256i pixels = _mm256_loadu_si256((__m256i*) & data[i]);

        __m256i maskForInversation = _mm256_cmpgt_epi8(_mm256_subs_epu8(pixels, _mm256_set1_epi8(128)), _mm256_setzero_si256());
        __m256i inverted = _mm256_sub_epi8(_mm256_set1_epi8(255), pixels);

        __m256i result = _mm256_blendv_epi8(pixels, inverted, maskForInversation);
        if (channels == 4) {
            result = _mm256_blendv_epi8(pixels, result, mask);
        }
        _mm256_storeu_si256((__m256i*) & data[i], result);
    }
    for (; i < width * height * channels; i++) {
        if (data[i] > 128) {
            data[i] = 255 - data[i];
        }
    }
}

__m256i threesum(__m256i a, __m256i b, __m256i c) {
    __m256i temp = _mm256_add_epi32(a, b);
    _mm256_add_epi32(c, temp);
    return temp;
}

void sepiaVector(unsigned char* data, int width, int height, int channels) {
    __m256i r1 = _mm256_set1_epi32((int)(0.393f * 256));
    __m256i r2 = _mm256_set1_epi32((int)(0.769f * 256));
    __m256i r3 = _mm256_set1_epi32((int)(0.189f * 256));
    __m256i g1 = _mm256_set1_epi32((int)(0.349f * 256));
    __m256i g2 = _mm256_set1_epi32((int)(0.686f * 256));
    __m256i g3 = _mm256_set1_epi32((int)(0.168f * 256));
    __m256i b1 = _mm256_set1_epi32((int)(0.272f * 256));
    __m256i b2 = _mm256_set1_epi32((int)(0.534f * 256));
    __m256i b3 = _mm256_set1_epi32((int)(0.131f * 256));
    __m256i maxValue = _mm256_set1_epi32(255);
    int i = 0;
    __m256i mask;
    if (channels == 4) {
        mask = _mm256_set_epi8(
            0, -1, -1, -1,
            0, -1, -1, -1,
            0, -1, -1, -1,
            0, -1, -1, -1,
            0, -1, -1, -1,
            0, -1, -1, -1,
            0, -1, -1, -1,
            0, -1, -1, -1
        );
        for (; i + 32 <= width * height * channels; i += 32) {
            __m256i pixels = _mm256_loadu_si256((__m256i*) & data[i]);
            __m256i red = _mm256_and_si256(pixels, maxValue);
            __m256i green = _mm256_and_si256(_mm256_srli_epi32(pixels, 8), maxValue);
            __m256i blue = _mm256_and_si256(_mm256_srli_epi32(pixels, 16), maxValue);
            __m256i rs = _mm256_min_epi32(_mm256_srli_epi32(threesum(_mm256_mullo_epi32(red, r1), _mm256_mullo_epi32(green, r2), _mm256_mullo_epi32(blue, r3)), 8), maxValue);
            __m256i gs = _mm256_min_epi32(_mm256_srli_epi32(threesum(_mm256_mullo_epi32(red, g1), _mm256_mullo_epi32(green, g2), _mm256_mullo_epi32(blue, g3)), 8), maxValue);
            __m256i bs = _mm256_min_epi32(_mm256_srli_epi32(threesum(_mm256_mullo_epi32(red, b1), _mm256_mullo_epi32(green, b2), _mm256_mullo_epi32(blue, b3)), 8), maxValue);
            __m256i packed = _mm256_or_si256(_mm256_or_si256(_mm256_slli_epi32(rs, 0), _mm256_slli_epi32(gs, 8)), _mm256_slli_epi32(bs, 16));

            __m256i result = _mm256_blendv_epi8(pixels, packed, mask);
            _mm256_storeu_si256((__m256i*) & data[i], result);
        }
    }
}

int main() {

    int l = 3;
    for (int i = l - 2; i > 0; i--) {
        std::cout << i << " ";
    }
}