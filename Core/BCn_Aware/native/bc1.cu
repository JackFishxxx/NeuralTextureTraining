#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// Keep Python/autograd headers out of NVCC's Windows translation unit.

// One half warp owns a BC1 block. Its 16 lanes own the block's texels;
// endpoint and mode gradients reduce within that half warp without atomics.
__device__ void palettes_for(const float* endpoints, float palette[2][4][4], int& first) {
    int codes[2][3], words[2];
    for (int e = 0; e < 2; ++e) {
        for (int c = 0; c < 3; ++c) {
            float limit = c == 1 ? 63.f : 31.f;
            codes[e][c] = __float2int_rn(fminf(fmaxf(endpoints[e*3+c], 0.f), limit));
        }
        words[e] = (codes[e][0] << 11) | (codes[e][1] << 5) | codes[e][2];
    }
    first = words[0] <= words[1] ? 0 : 1;  // Stable ordering for equal words.
    int low = words[first], high = words[1-first];
    int opaque_high = min(max(high, low+1), 65535);
    int opaque_low = max(min(low, opaque_high-1), 0);
    for (int c = 0; c < 3; ++c) {
        int shift = c == 0 ? 11 : c == 1 ? 5 : 0;
        int mask = c == 1 ? 63 : 31;
        int v[4] = {(low >> shift) & mask, (high >> shift) & mask,
                    (opaque_high >> shift) & mask, (opaque_low >> shift) & mask};
        for (int i = 0; i < 4; ++i)
            v[i] = c == 1 ? (v[i] << 2) | (v[i] >> 4) : (v[i] << 3) | (v[i] >> 2);
        palette[0][0][c] = v[0] * (1.f/255.f);
        palette[0][1][c] = v[1] * (1.f/255.f);
        palette[0][2][c] = ((v[0]+v[1])/2) * (1.f/255.f);
        palette[0][3][c] = 0.f;
        palette[1][0][c] = v[2] * (1.f/255.f);
        palette[1][1][c] = v[3] * (1.f/255.f);
        palette[1][2][c] = ((2*v[2]+v[3])/3) * (1.f/255.f);
        palette[1][3][c] = ((v[2]+2*v[3])/3) * (1.f/255.f);
    }
    for (int m = 0; m < 2; ++m)
        for (int s = 0; s < 4; ++s) palette[m][s][3] = (m == 0 && s == 3) ? 0.f : 1.f;
}

__device__ int selector_for(const float* weights) {
    int selected = 0;
    for (int i = 1; i < 4; ++i) if (weights[i] > weights[selected]) selected = i;
    return selected;
}

__global__ void decode_kernel(const float* endpoints, const float* weights, const float* modes,
                              float* image, int height, int width, int channels, int blocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x, block = tid / 16, pixel = tid % 16;
    if (block >= blocks) return;
    int bw = (width+3)/4, bh = (height+3)/4;
    int surface = block / (bw*bh), local = block % (bw*bh);
    int y = local/bw*4 + pixel/4, x = local%bw*4 + pixel%4;
    if (y >= height || x >= width) return;
    float palette[2][4][4]; int first;
    palettes_for(endpoints+block*6, palette, first);
    int mode = modes[block*2+1] > modes[block*2] ? 1 : 0;
    int selected = selector_for(weights+block*64+pixel*4);
    for (int c = 0; c < 4; ++c)
        image[(y*width+x)*channels+surface*4+c] = palette[mode][selected][c];
}

__global__ void backward_kernel(const float* gradient, const float* endpoints, const float* weights,
    const float* modes, const float* temperature, float* ge, float* gw, float* gm,
    int height, int width, int channels, int blocks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x, block = tid / 16, pixel = tid % 16;
    if (block >= blocks) return;
    int bw = (width+3)/4, bh = (height+3)/4;
    int surface = block/(bw*bh), local = block%(bw*bh);
    int y = local/bw*4+pixel/4, x = local%bw*4+pixel%4;
    float g[4] = {};
    if (y < height && x < width)
        for (int c = 0; c < 4; ++c) g[c] = gradient[(y*width+x)*channels+surface*4+c];
    float palette[2][4][4]; int first;
    palettes_for(endpoints+block*6, palette, first);
    const float* w = weights+block*64+pixel*4;
    const float* m = modes+block*2;
    int mode = m[1] > m[0] ? 1 : 0, selected = selector_for(w);
    float temp = fmaxf(temperature[0], .1f);
    float mp[2], sp[4], ms = 0.f, ss = 0.f;
    float mm = fmaxf(m[0]/temp, m[1]/temp), wm = w[0]/temp;
    for (int i = 1; i < 4; ++i) wm = fmaxf(wm, w[i]/temp);
    for (int i = 0; i < 2; ++i) { mp[i] = expf(m[i]/temp-mm); ms += mp[i]; }
    for (int i = 0; i < 4; ++i) { sp[i] = expf(w[i]/temp-wm); ss += sp[i]; }
    for (int i = 0; i < 2; ++i) mp[i] /= ms;
    for (int i = 0; i < 4; ++i) sp[i] /= ss;
    float selector_scores[4] = {}, mode_scores[2] = {};
    for (int s = 0; s < 4; ++s) {
        float score[2] = {};
        for (int c = 0; c < 4; ++c) {
            selector_scores[s] += g[c]*(palette[0][s][c]*mp[0] + palette[1][s][c]*mp[1]);
            for (int i = 0; i < 2; ++i) score[i] += g[c]*palette[i][s][c];
        }
        for (int i = 0; i < 2; ++i) mode_scores[i] += score[i]*sp[s];
    }
    float expectation = 0.f;
    for (int s = 0; s < 4; ++s) expectation += selector_scores[s]*sp[s];
    for (int s = 0; s < 4; ++s)
        gw[block*64+pixel*4+s] = sp[s]*(selector_scores[s]-expectation)/temp;
    float mode_expectation = mode_scores[0]*mp[0] + mode_scores[1]*mp[1];
    // All lanes in each half warp participate, including padded edge texels.
    unsigned mask = __activemask();
    for (int i = 0; i < 2; ++i) {
        float value = mp[i]*(mode_scores[i]-mode_expectation)/temp;
        for (int offset = 8; offset > 0; offset /= 2)
            value += __shfl_down_sync(mask, value, offset, 16);
        if (pixel == 0) gm[block*2+i] = value;
    }
    float low_coefficient = 0.f, high_coefficient = 0.f;
    if (mode == 0) {
        low_coefficient = selected == 0 ? 1.f : selected == 2 ? .5f : 0.f;
        high_coefficient = selected == 1 ? 1.f : selected == 2 ? .5f : 0.f;
    } else {
        low_coefficient = selected == 1 ? 1.f : selected == 2 ? 1.f/3.f : selected == 3 ? 2.f/3.f : 0.f;
        high_coefficient = selected == 0 ? 1.f : selected == 2 ? 2.f/3.f : selected == 3 ? 1.f/3.f : 0.f;
    }
    for (int e = 0; e < 2; ++e) {
        float coefficient = e == first ? low_coefficient : high_coefficient;
        for (int c = 0; c < 3; ++c) {
            float value = g[c]*coefficient;
            for (int offset = 8; offset > 0; offset /= 2)
                value += __shfl_down_sync(mask, value, offset, 16);
            if (pixel == 0) {
                float raw = endpoints[block*6+e*3+c], limit = c == 1 ? 63.f : 31.f;
                float clamp_gradient = raw < 0.f || raw > limit ? 0.f : raw == limit ? .5f : 1.f;
                ge[block*6+e*3+c] = value * (1.f/limit) * clamp_gradient;
            }
        }
    }
}

void check_inputs(const at::Tensor& endpoints, const at::Tensor& weights,
                  const at::Tensor& modes, int64_t h, int64_t w, int64_t c) {
    TORCH_CHECK(h > 0 && w > 0 && (c == 4 || c == 8), "Invalid BC1 image shape");
    int64_t blocks = ((h+3)/4)*((w+3)/4)*(c/4);
    for (const auto& tensor : {endpoints, weights, modes}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == at::kFloat && tensor.is_contiguous(),
                    "BC1 CUDA inputs must be contiguous CUDA float32 tensors");
        TORCH_CHECK(tensor.device() == endpoints.device(), "BC1 inputs must use the same CUDA device");
    }
    TORCH_CHECK(endpoints.sizes() == at::IntArrayRef({blocks,2,3}) &&
                weights.sizes() == at::IntArrayRef({blocks,16,4}) &&
                modes.sizes() == at::IntArrayRef({blocks,2}), "Invalid BC1 parameter shapes");
}

at::Tensor bc1_decode_cuda(at::Tensor endpoints, at::Tensor weights,
    at::Tensor modes, int64_t h, int64_t w, int64_t c) {
    check_inputs(endpoints, weights, modes, h, w, c);
    const c10::cuda::CUDAGuard guard(endpoints.device());
    auto image = at::empty({h,w,c}, endpoints.options());
    int blocks = endpoints.size(0);
    decode_kernel<<<(blocks*16+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(
        endpoints.data_ptr<float>(), weights.data_ptr<float>(), modes.data_ptr<float>(),
        image.data_ptr<float>(), h, w, c, blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return image;
}

std::vector<at::Tensor> bc1_backward_cuda(at::Tensor gradient, at::Tensor endpoints,
    at::Tensor weights, at::Tensor modes, at::Tensor temperature, int64_t h, int64_t w, int64_t c) {
    check_inputs(endpoints, weights, modes, h, w, c);
    TORCH_CHECK(gradient.is_cuda() && gradient.device() == endpoints.device() &&
        gradient.scalar_type() == at::kFloat && gradient.is_contiguous() &&
        gradient.sizes() == at::IntArrayRef({h,w,c}), "Invalid BC1 output gradient");
    TORCH_CHECK(temperature.device() == endpoints.device() && temperature.scalar_type() == at::kFloat &&
        temperature.numel() == 1, "Invalid BC1 temperature");
    const c10::cuda::CUDAGuard guard(endpoints.device());
    auto ge = at::empty_like(endpoints), gw = at::empty_like(weights), gm = at::empty_like(modes);
    int blocks = endpoints.size(0);
    backward_kernel<<<(blocks*16+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(
        gradient.data_ptr<float>(), endpoints.data_ptr<float>(), weights.data_ptr<float>(),
        modes.data_ptr<float>(), temperature.data_ptr<float>(), ge.data_ptr<float>(),
        gw.data_ptr<float>(), gm.data_ptr<float>(), h, w, c, blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {ge,gw,gm};
}

// Compute only the two probabilities used by binary usage; never materialize
// the full-grid softmax tensors or their elementwise autograd intermediates.
__device__ void usage_probabilities(const float* weights, const float* modes, float temp,
                                    float sp[4], float mp[2]) {
    float wm = weights[0]/temp, mm = fmaxf(modes[0]/temp,modes[1]/temp), sum = 0.f;
    for (int i = 1; i < 4; ++i) wm = fmaxf(wm,weights[i]/temp);
    for (int i = 0; i < 4; ++i) { sp[i] = expf(weights[i]/temp-wm); sum += sp[i]; }
    for (int i = 0; i < 4; ++i) sp[i] /= sum;
    mp[0] = expf(modes[0]/temp-mm); mp[1] = expf(modes[1]/temp-mm);
    sum = mp[0]+mp[1]; mp[0] /= sum; mp[1] /= sum;
}

__device__ float usage_reduce(float value, float* shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x+stride];
        __syncthreads();
    }
    return shared[0];
}

__global__ void usage_partials_kernel(const float* weights, const float* modes, const float* temperature,
                                      float* partials, int pixels, int chunks) {
    int pixel = blockIdx.x*blockDim.x+threadIdx.x;
    int global_pixel = blockIdx.y*pixels+pixel, block = global_pixel/16;
    float hard = 0.f, confidence = 0.f;
    if (pixel < pixels) {
        const float* w = weights+global_pixel*4;
        const float* m = modes+block*2;
        float sp[4],mp[2];
        usage_probabilities(w,m,fmaxf(temperature[0],.1f),sp,mp);
        float zero = mp[0]*sp[3];
        hard = m[0] >= m[1] && selector_for(w) == 3 ? 1.f : 0.f;
        confidence = zero*(1.f-zero);
    }
    __shared__ float shared[256];
    float occupancy_sum = usage_reduce(hard,shared);
    if (threadIdx.x == 0) partials[(blockIdx.y*chunks+blockIdx.x)*2] = occupancy_sum;
    __syncthreads();
    float confidence_sum = usage_reduce(confidence,shared);
    if (threadIdx.x == 0) partials[(blockIdx.y*chunks+blockIdx.x)*2+1] = confidence_sum;
}

__global__ void usage_finish_kernel(const float* partials, float* loss, float* occupancy_gradient,
                                    int pixels, int chunks, int surfaces) {
    __shared__ float shared[256];
    float total = 0.f;
    for (int surface = 0; surface < surfaces; ++surface) {
        float hard = 0.f, confidence = 0.f;
        for (int i = threadIdx.x; i < chunks; i += blockDim.x) {
            hard += partials[(surface*chunks+i)*2];
            confidence += partials[(surface*chunks+i)*2+1];
        }
        float occupancy = usage_reduce(hard,shared)/pixels;
        __syncthreads();
        float mean_confidence = usage_reduce(confidence,shared)/pixels;
        if (threadIdx.x == 0) {
            float below = fmaxf(.1f-occupancy,0.f), above = fmaxf(occupancy-.9f,0.f);
            total += (below*below+above*above+.01f*mean_confidence)/surfaces;
            occupancy_gradient[surface] = (2.f*above-2.f*below)/(float(pixels)*surfaces);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) loss[0] = total;
}

__global__ void usage_backward_kernel(const float* gradient, const float* weights, const float* modes,
    const float* temperature, const float* occupancy_gradient, float* gw, float* gm, int blocks, int surfaces) {
    int global_pixel = blockIdx.x*blockDim.x+threadIdx.x;
    int block = global_pixel/16, pixel = global_pixel%16;
    if (block >= blocks) return;
    int pixels = blocks/surfaces*16, surface = global_pixel/pixels;
    float sp[4],mp[2],temp = fmaxf(temperature[0],.1f);
    usage_probabilities(weights+global_pixel*4,modes+block*2,temp,sp,mp);
    float zero = mp[0]*sp[3];
    float dz = gradient[0]*(occupancy_gradient[surface]+.01f*(1.f-2.f*zero)/(float(pixels)*surfaces));
    for (int i = 0; i < 4; ++i)
        gw[global_pixel*4+i] = (dz*mp[0]*sp[3])*((i == 3 ? 1.f : 0.f)-sp[i])/temp;
    unsigned mask = __activemask();
    for (int i = 0; i < 2; ++i) {
        float value = (dz*sp[3]*mp[0])*((i == 0 ? 1.f : 0.f)-mp[i])/temp;
        for (int offset = 8; offset > 0; offset /= 2)
            value += __shfl_down_sync(mask,value,offset,16);
        if (pixel == 0) gm[block*2+i] = value;
    }
}

void check_usage_inputs(const at::Tensor& weights, const at::Tensor& modes,
                        const at::Tensor& temperature, int64_t surfaces) {
    TORCH_CHECK((surfaces == 1 || surfaces == 2) && weights.dim() == 3 && weights.size(0) > 0 &&
        weights.size(0)%surfaces == 0 && weights.size(1) == 16 && weights.size(2) == 4 &&
        modes.sizes() == at::IntArrayRef({weights.size(0),2}) && temperature.numel() == 1,
        "Invalid BC1 usage parameter shapes");
    for (const auto& tensor : {weights,modes,temperature})
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == weights.device() &&
            tensor.scalar_type() == at::kFloat && tensor.is_contiguous(),
            "BC1 usage inputs must be contiguous CUDA float32 tensors on the same device");
}

std::vector<at::Tensor> bc1_usage_cuda(at::Tensor weights, at::Tensor modes,
                                     at::Tensor temperature, int64_t surfaces) {
    check_usage_inputs(weights,modes,temperature,surfaces);
    const c10::cuda::CUDAGuard guard(weights.device());
    int pixels = weights.size(0)/surfaces*16, chunks = (pixels+255)/256;
    auto partials = at::empty({surfaces,chunks,2},weights.options());
    auto loss = at::empty({},weights.options()), occupancy_gradient = at::empty({surfaces},weights.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    usage_partials_kernel<<<dim3(chunks,surfaces),256,0,stream>>>(weights.data_ptr<float>(),modes.data_ptr<float>(),
        temperature.data_ptr<float>(),partials.data_ptr<float>(),pixels,chunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    usage_finish_kernel<<<1,256,0,stream>>>(partials.data_ptr<float>(),loss.data_ptr<float>(),
        occupancy_gradient.data_ptr<float>(),pixels,chunks,surfaces);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {loss,occupancy_gradient};
}

std::vector<at::Tensor> bc1_usage_backward_cuda(at::Tensor gradient, at::Tensor weights,
    at::Tensor modes, at::Tensor temperature, at::Tensor occupancy_gradient, int64_t surfaces) {
    check_usage_inputs(weights,modes,temperature,surfaces);
    TORCH_CHECK(gradient.device() == weights.device() && gradient.scalar_type() == at::kFloat &&
        gradient.numel() == 1 && gradient.is_contiguous() && occupancy_gradient.device() == weights.device() &&
        occupancy_gradient.scalar_type() == at::kFloat && occupancy_gradient.is_contiguous() &&
        occupancy_gradient.sizes() == at::IntArrayRef({surfaces}), "Invalid BC1 usage backward inputs");
    const c10::cuda::CUDAGuard guard(weights.device());
    auto gw = at::empty_like(weights), gm = at::empty_like(modes);
    int blocks = weights.size(0);
    usage_backward_kernel<<<(blocks*16+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(
        gradient.data_ptr<float>(),weights.data_ptr<float>(),modes.data_ptr<float>(),temperature.data_ptr<float>(),
        occupancy_gradient.data_ptr<float>(),gw.data_ptr<float>(),gm.data_ptr<float>(),blocks,surfaces);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gw,gm};
}
