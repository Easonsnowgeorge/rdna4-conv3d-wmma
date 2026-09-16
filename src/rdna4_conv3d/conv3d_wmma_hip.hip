// gfx1201 implicit-GEMM conv3d fprop, 3x3x3 stride-1, fp16 in / fp32 acc / fp16 out.
// Inner product is rocWMMA 16x16x16 (v_wmma_f32_16x16x16_f16_w32_gfx12).
//
// Input  : NDHWC (PyTorch channels_last_3d)
// Weight : packed (27*Cin, Cout), Cin fastest per tap
// Output : NDHWC
//
// Block tile 64x64, 4 waves (128 threads). Each wave owns a 32x32 quadrant
// made of four 16x16 WMMA fragments. K-tile is 16 to match WMMA_K.

#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

namespace {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int THREADS = 128;

__global__ void conv3d_k3_wmma_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    __half* __restrict__ y,
    const __half* __restrict__ bias,
    int N, int D, int H, int W, int Cin, int Cout,
    int OD, int OH, int OW,
    int pad_d, int pad_h, int pad_w,
    long sxn, long sxd, long sxh, long sxw, long sxc,
    long swk, long swn,
    long syn, long syd, long syh, long syw, long syc,
    int has_bias)
{
    using namespace rocwmma;
    using FragA = fragment<matrix_a, 16, 16, 16, __half, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, __half, row_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;

    const int gemm_m = N * OD * OH * OW;
    const int gemm_n = Cout;
    const int num_pid_n = (gemm_n + BN - 1) / BN;
    const int pid_m = blockIdx.x / num_pid_n;
    const int pid_n = blockIdx.x % num_pid_n;
    const int tid = threadIdx.x;
    const int wave = tid / 32;

    const int m0 = pid_m * BM;
    const int n0 = pid_n * BN;
    const int wave_m = (wave / 2) * 32;
    const int wave_n = (wave % 2) * 32;

    __shared__ __half As[BM * BK];
    __shared__ __half Bs[BK * BN];
    __shared__ __half Cs[BM * BN];

    FragC c00, c01, c10, c11;
    fill_fragment(c00, 0.0f);
    fill_fragment(c01, 0.0f);
    fill_fragment(c10, 0.0f);
    fill_fragment(c11, 0.0f);

    const int gemm_k = 27 * Cin;

    for (int k0 = 0; k0 < gemm_k; k0 += BK) {
        for (int i = tid; i < BM * BK; i += THREADS) {
            int m = i / BK;
            int kk = i % BK;
            int gm = m0 + m;
            int gk = k0 + kk;
            __half v = __float2half(0.f);
            if (gm < gemm_m && gk < gemm_k) {
                int ow = gm % OW;
                int t = gm / OW;
                int oh = t % OH;
                t = t / OH;
                int od = t % OD;
                int n = t / OD;
                int c = gk % Cin;
                int tap = gk / Cin;
                int kw = tap % 3;
                int kh = (tap / 3) % 3;
                int kd = tap / 9;
                int id = od + kd - pad_d;
                int ih = oh + kh - pad_h;
                int iw = ow + kw - pad_w;
                if ((unsigned)id < (unsigned)D && (unsigned)ih < (unsigned)H &&
                    (unsigned)iw < (unsigned)W) {
                    v = x[n * sxn + id * sxd + ih * sxh + iw * sxw + c * sxc];
                }
            }
            As[m * BK + kk] = v;
        }
        for (int i = tid; i < BK * BN; i += THREADS) {
            int kk = i / BN;
            int nn = i % BN;
            int gk = k0 + kk;
            int gn = n0 + nn;
            __half v = __float2half(0.f);
            if (gk < gemm_k && gn < gemm_n) {
                v = w[gk * swk + gn * swn];
            }
            Bs[kk * BN + nn] = v;
        }
        __syncthreads();

        FragA a0, a1;
        FragB b0, b1;
        load_matrix_sync(a0, As + (wave_m + 0) * BK, BK);
        load_matrix_sync(a1, As + (wave_m + 16) * BK, BK);
        load_matrix_sync(b0, Bs + wave_n, BN);
        load_matrix_sync(b1, Bs + wave_n + 16, BN);
        mma_sync(c00, a0, b0, c00);
        mma_sync(c01, a0, b1, c01);
        mma_sync(c10, a1, b0, c10);
        mma_sync(c11, a1, b1, c11);
        __syncthreads();
    }

    store_matrix_sync(Cs + (wave_m + 0) * BN + (wave_n + 0), c00, BN, mem_row_major);
    store_matrix_sync(Cs + (wave_m + 0) * BN + (wave_n + 16), c01, BN, mem_row_major);
    store_matrix_sync(Cs + (wave_m + 16) * BN + (wave_n + 0), c10, BN, mem_row_major);
    store_matrix_sync(Cs + (wave_m + 16) * BN + (wave_n + 16), c11, BN, mem_row_major);
    __syncthreads();

    for (int i = tid; i < BM * BN; i += THREADS) {
        int m = i / BN;
        int nn = i % BN;
        int gm = m0 + m;
        int gn = n0 + nn;
        if (gm < gemm_m && gn < gemm_n) {
            int ow = gm % OW;
            int t = gm / OW;
            int oh = t % OH;
            t = t / OH;
            int od = t % OD;
            int n = t / OD;
            float acc = __half2float(Cs[m * BN + nn]);
            if (has_bias) acc += __half2float(bias[gn]);
            y[n * syn + od * syd + oh * syh + ow * syw + gn * syc] = __float2half(acc);
        }
    }
}

}  // namespace

torch::Tensor hip_conv3d_k3_fwd(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor bias,
    int pad_d, int pad_h, int pad_w)
{
    TORCH_CHECK(x.is_cuda() && w_packed.is_cuda(), "HIP conv3d expects GPU tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat16, "fp16 only");
    TORCH_CHECK(x.dim() == 5, "expected NCDHW / NDHWC 5D");

    const int N = x.size(0);
    const int Cin = x.size(1);
    const int D = x.size(2);
    const int H = x.size(3);
    const int W = x.size(4);
    const int Cout = w_packed.size(1);
    const int OD = D + 2 * pad_d - 2;
    const int OH = H + 2 * pad_h - 2;
    const int OW = W + 2 * pad_w - 2;

    auto y = torch::empty({N, Cout, OD, OH, OW},
                          x.options().memory_format(at::MemoryFormat::ChannelsLast3d));

    const int gemm_m = N * OD * OH * OW;
    const int gemm_n = Cout;
    const int grid = ((gemm_m + BM - 1) / BM) * ((gemm_n + BN - 1) / BN);
    const __half* bias_ptr = nullptr;
    int has_bias = 0;
    if (bias.defined() && bias.numel() == Cout) {
        bias_ptr = reinterpret_cast<const __half*>(bias.data_ptr<at::Half>());
        has_bias = 1;
    }

    hipLaunchKernelGGL(
        conv3d_k3_wmma_kernel, dim3(grid), dim3(THREADS), 0, 0,
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(w_packed.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        bias_ptr,
        N, D, H, W, Cin, Cout, OD, OH, OW, pad_d, pad_h, pad_w,
        (long)x.stride(0), (long)x.stride(2), (long)x.stride(3), (long)x.stride(4), (long)x.stride(1),
        (long)w_packed.stride(0), (long)w_packed.stride(1),
        (long)y.stride(0), (long)y.stride(2), (long)y.stride(3), (long)y.stride(4), (long)y.stride(1),
        has_bias);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd_k3", &hip_conv3d_k3_fwd, "implicit-GEMM conv3d 3x3x3 WMMA fprop");
}
