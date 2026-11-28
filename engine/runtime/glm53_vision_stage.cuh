// GLM-5.3 image input in the prefill (GR8_GLM53_VISION / GR8_GLM53_VISION_PATH): before each prefill launch, every
// image whose span overlaps the launch's rows runs through the GLM-5.3 vision tower (glm53_vision.cuh), and its
// overlapping rows (including row_break_embed at the end of each merged row) are copied into a device pool. The
// launch's embed (prefill.cuh pf_embed -> glue::k_embed_norm_vis) then reads those rows into the FP32 residual stream
// in place of the <|image|> embedding row. A text-only launch stages nothing and runs the unchanged k_embed_norm.
//
// When P.perm is true (GR8_PF_OWNPERM=1), pf_inputs permutes row j = o[j] via pf_perm_order(rows, P.f.nr); stage()
// applies the exact same permutation to P.vis_row so each permuted row maps to its original image pool row.
#pragma once
#if !G6_MODEL_FLASH && !G6_MODEL_DS41
#include "glm53_vision.cuh"
#include "image_input.h"
#include "prefill.cuh"

namespace g6rt
{

static_assert(HID == glm53v::OD_OUT, "the GLM-5.3 vision tower's rows are the text model's embedding rows");

class Glm53Vision
{
public:
    // Tower weights and workspace, a pool of pool_rows image rows (every lane of one quantum) and one image's staging
    // rows, all allocated at boot before the KV pool is sized.
    void boot(const std::string& path, int pool_rows)
    {
        tower_.load(path);
        tower_.ensure_workspace();
        pool_rows_ = pool_rows;
        G6RT_CK(cudaMalloc(&pool_, (size_t)pool_rows * HID * sizeof(__nv_bfloat16)));
        G6RT_CK(cudaMalloc(&stage_, (size_t)glm53v::MAX_SPAN * HID * sizeof(__nv_bfloat16)));
    }
    size_t device_bytes() const
    {
        return tower_.w.bytes + (size_t)glm53v::VL * 2 * glm53v::VI * 2 + tower_.workspace_bytes() +
               ((size_t)pool_rows_ + glm53v::MAX_SPAN) * HID * sizeof(__nv_bfloat16);
    }
    // A prefill lane that may launch image rows: allocates its row map and binds the shared pool.
    void attach(Prefill& P)
    {
        G6RT_CK(cudaMalloc(&P.vis_row, (size_t)P.Mmax * sizeof(int)));
        P.vis_pool = pool_;
        P.vis_n = 0;
    }
    // Before a launch of P's `rows` rows at prompt positions first.. on st: run the tower on overlapping images, copy
    // their rows into the pool from pool_row on, write P's row map (permuted if P.perm) and set P.vis_n.
    int stage(Prefill& P, const std::vector<ImageIn>& images, int first, int rows, int& pool_row, cudaStream_t st,
              int rank, unsigned long long request)
    {
        P.vis_n = 0;
        if (images.empty()) return 0;
        if (rows < 1 || rows > P.Mmax) die("GLM-5.3 vision: launch rows outside 1..Mmax");
        map_.assign((size_t)rows, -1);
        int n = 0;
        for (size_t k = 0; k < images.size(); ++k) {
            const ImageIn& im = images[k];
            const int a = std::max(im.pos, first), b = std::min(im.pos + im.span, first + rows);
            if (a >= b) continue;
            if (pool_row + (b - a) > pool_rows_) die("GLM-5.3 vision: the image rows of one prefill quantum overflow the pool");
            const bool reuse = last_ == im.rgb && last_h_ == im.vit_h && last_w_ == im.vit_w;
            if (!reuse) {
                const auto t0 = std::chrono::steady_clock::now();
                last_.reset();
                try { tower_.encode(im.rgb->data(), im.vit_h, im.vit_w, stage_, st); }
                catch (const std::exception& e) { die(e.what()); }
                last_ = im.rgb; last_h_ = im.vit_h; last_w_ = im.vit_w;
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (rank == 0)
                    fprintf(stderr, "# glm53 vision: request %llu image %zu: %dx%d patches, %d span tokens, tower %.1f ms\n",
                            request, k, im.vit_h, im.vit_w, im.span, ms);
            }
            G6RT_CK(cudaMemcpyAsync(pool_ + (size_t)pool_row * HID, stage_ + (size_t)(a - im.pos) * HID,
                                    (size_t)(b - a) * HID * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, st));
            for (int p = a; p < b; ++p) map_[(size_t)(p - first)] = pool_row + (p - a);
            pool_row += b - a;
            n += b - a;
        }
        if (n) {
            if (P.perm) {
                const std::vector<int> o = pf_perm_order(rows, P.f.nr);
                perm_map_.resize((size_t)rows);
                for (int j = 0; j < rows; ++j) perm_map_[(size_t)j] = map_[(size_t)o[(size_t)j]];
                G6RT_CK(cudaMemcpyAsync(P.vis_row, perm_map_.data(), (size_t)rows * sizeof(int), cudaMemcpyHostToDevice, st));
            } else {
                G6RT_CK(cudaMemcpyAsync(P.vis_row, map_.data(), (size_t)rows * sizeof(int), cudaMemcpyHostToDevice, st));
            }
            G6RT_CK(cudaStreamSynchronize(st));   // the next lane's stage reuses map_ / perm_map_
            P.vis_n = n;
        }
        return n;
    }

private:
    glm53v::Tower tower_;
    __nv_bfloat16* pool_ = nullptr;
    __nv_bfloat16* stage_ = nullptr;   // last image's row-break-expanded rows: [MAX_SPAN][HID]
    int pool_rows_ = 0;
    std::shared_ptr<const std::vector<uint8_t>> last_;
    int last_h_ = 0, last_w_ = 0;
    std::vector<int> map_;
    std::vector<int> perm_map_;
};

}   // namespace g6rt
#endif
