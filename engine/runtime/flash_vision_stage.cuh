// GLM-5.3-Flash image input in the prefill (GR8_GLM53F_VISION, docs/FLASH-VISION.md): before each prefill launch, every
// image whose span overlaps the launch's rows runs through the vision tower (flash_vision.cuh), and its overlapping
// rows are copied to a device pool. The launch's embed (flash_prefill.cuh pf_embed_flash) then reads those rows in place
// of the <|image|> embedding. A text-only launch stages nothing and runs the text-only embed.
//
// The rows of a launch are consecutive positions of one session (pf_inputs_flash), so a launch row is its position
// minus the launch's first. The lanes of one quantum share the pool (pass the same row counter) and the compute stream.
// The tower's last output stays in the stage buffer, so an image that straddles two launches (two lanes, or two
// quanta) runs through the tower once.
#pragma once
#if G6_MODEL_FLASH
#include "flash_vision.cuh"
#include "image_input.h"

namespace g6rt
{

static_assert(HID == glm5v::OD, "the tower's rows are the text model's embedding rows");

class FlashVision
{
public:
    // the tower's weights and workspace, a pool of pool_rows image rows (every lane of one quantum) and one image's rows
    // of staging, all before the KV pool is sized
    void boot(const std::string& path, int pool_rows)
    {
        tower_.load(path);
        tower_.ensure_workspace();
        pool_rows_ = pool_rows;
        G6RT_CK(cudaMalloc(&pool_, (size_t)pool_rows * HID * sizeof(__nv_bfloat16)));
        G6RT_CK(cudaMalloc(&stage_, (size_t)glm5v::MAX_TOKENS * HID * sizeof(__nv_bfloat16)));
    }
    size_t device_bytes() const
    {
        return tower_.w.bytes + (size_t)glm5v::VL * 2 * glm5v::VI * 2 + tower_.workspace_bytes() +
               ((size_t)pool_rows_ + glm5v::MAX_TOKENS) * HID * sizeof(__nv_bfloat16);
    }
    // a prefill lane that may launch image rows: its row map, and the pool
    void attach(FlashPrefill& P)
    {
        G6RT_CK(cudaMalloc(&P.vis_row, (size_t)P.MP * sizeof(int)));
        P.vis_pool = pool_;
        P.vis_n = 0;
    }
    // Before a launch of P's `rows` rows at prompt positions first.. on st: run the tower on the images that overlap them,
    // copy their rows into the pool from pool_row on, write P's row map and set P.vis_n (0 when no row is an image row).
    // The caller clears P.vis_n after the launch. Returns the number of image rows.
    int stage(FlashPrefill& P, const std::vector<ImageIn>& images, int first, int rows, int& pool_row, cudaStream_t st,
              int rank, unsigned long long request)
    {
        P.vis_n = 0;
        if (images.empty()) return 0;
        if (rows < 1 || rows > P.MP) die("Flash vision: launch rows outside 1..MP");
        map_.assign((size_t)rows, -1);
        int n = 0;
        for (size_t k = 0; k < images.size(); ++k) {
            const ImageIn& im = images[k];
            const int a = std::max(im.pos, first), b = std::min(im.pos + im.span, first + rows);
            if (a >= b) continue;
            if (pool_row + (b - a) > pool_rows_) die("Flash vision: the image rows of one prefill quantum overflow the pool");
            const bool reuse = last_ == im.rgb && last_h_ == im.vit_h && last_w_ == im.vit_w;
            if (!reuse) {
                const auto t0 = std::chrono::steady_clock::now();
                last_.reset();
                try { tower_.encode(im.rgb->data(), im.vit_h, im.vit_w, stage_, st); }
                catch (const std::exception& e) { die(e.what()); }
                last_ = im.rgb; last_h_ = im.vit_h; last_w_ = im.vit_w;
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (rank == 0)
                    fprintf(stderr, "# flash vision: request %llu image %zu: %dx%d patches, %d tokens, tower %.1f ms\n", request, k,
                            im.vit_h, im.vit_w, im.span, ms);
            }
            G6RT_CK(cudaMemcpyAsync(pool_ + (size_t)pool_row * HID, stage_ + (size_t)(a - im.pos) * HID,
                                    (size_t)(b - a) * HID * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, st));
            for (int p = a; p < b; ++p) map_[(size_t)(p - first)] = pool_row + (p - a);
            pool_row += b - a;
            n += b - a;
        }
        if (n) {
            G6RT_CK(cudaMemcpyAsync(P.vis_row, map_.data(), (size_t)rows * sizeof(int), cudaMemcpyHostToDevice, st));
            G6RT_CK(cudaStreamSynchronize(st));   // the next lane's stage reuses map_
            P.vis_n = n;
        }
        return n;
    }

private:
    glm5v::Tower tower_;
    __nv_bfloat16* pool_ = nullptr;
    __nv_bfloat16* stage_ = nullptr;   // the last image's rows: [span][HID]
    int pool_rows_ = 0;
    std::shared_ptr<const std::vector<uint8_t>> last_;   // whose rows stage_ holds (kept alive, so the pointer stays unique)
    int last_h_ = 0, last_w_ = 0;
    std::vector<int> map_;
};

}   // namespace g6rt
#endif
