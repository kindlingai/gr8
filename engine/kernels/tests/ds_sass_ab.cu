// SASS A/B + compile check of the MoE kernels. Takes the address of every default (M <= 8) instance so nvcc compiles them; with
// -DAB_MT=1 also the multi-tile instances. No GPU needed:
//   nvcc -O2 -std=c++17 -gencode=arch=compute_121a,code=sm_121a -cubin -Xptxas -v [-DAB_MT=1] -o ab.cubin ds_sass_ab.cu && cuobjdump -sass ab.cubin
// The M <= 8 instances must give the same SASS with and without -DAB_MT=1.
#include "../ds_shared.cuh"

void* const g_ab_kernels[] = {
    (void*)dsr::k_gate_logits<384>, (void*)dsr::k_gate_logits<128>,
    (void*)dsr::k_plan<384, 6>, (void*)dsr::k_plan<128, 3>,
    (void*)mx4::build_xfrag,
    (void*)mx4::k_fc1<5120, 576, 8, 1, 2, false>, (void*)mx4::k_fc2<576, 5120, 8, 2, false>,
    (void*)dsh::k_sh_fc1<5120, 576, 8, 1, 2>, (void*)dsh::k_sh_fc2<576, 5120, 8, 3>,
    (void*)dsh::k_combine<5120, 128, gv::O_BF16, 6>, (void*)dsh::k_combine<5120, 128, gv::O_BF16, 3>, (void*)dsh::k_combine<5120, 128, gv::O_F32, 6>,
#ifdef AB_MT
    (void*)dsr::k_gate_logits_mt<384>, (void*)dsr::k_gate_logits_mt<128>,
    (void*)dsr::k_plan_mt<384, 6, 32>, (void*)dsr::k_plan_mt<128, 3, 32>,
    (void*)mx4::build_xfrag_mt<0>,
    (void*)mx4::k_fc1_mt<5120, 576, 8, 1, 2, 2>, (void*)mx4::k_fc1_mt<5120, 576, 8, 1, 2, 3>, (void*)mx4::k_fc1_mt<5120, 576, 8, 1, 2, 4>,
    (void*)mx4::k_fc2_mt<576, 5120, 8, 2, 2>, (void*)mx4::k_fc2_mt<576, 5120, 8, 2, 3>, (void*)mx4::k_fc2_mt<576, 5120, 8, 2, 4>,
    (void*)dsh::k_sh_fc1_mt<5120, 576, 8, 1, 2, 2>, (void*)dsh::k_sh_fc1_mt<5120, 576, 8, 1, 2, 3>, (void*)dsh::k_sh_fc1_mt<5120, 576, 8, 1, 2, 4>,
    (void*)dsh::k_sh_fc2_mt<576, 5120, 8, 3, 2>, (void*)dsh::k_sh_fc2_mt<576, 5120, 8, 3, 3>, (void*)dsh::k_sh_fc2_mt<576, 5120, 8, 3, 4>,
    (void*)dsh::k_combine_mt<5120, 128, gv::O_BF16, 6, 32>, (void*)dsh::k_combine_mt<5120, 128, gv::O_BF16, 3, 32>,
#endif
};
#ifdef AB_PDL
// the PDL twins and PDLM = 1 instances: compile check only (no baseline)
void* const g_ab_pdl[] = {
    (void*)dsr::k_gate_logits<384, 1>, (void*)dsr::k_gate_logits<128, 1>, (void*)dsr::k_plan_p<384, 6>, (void*)dsr::k_plan_p<128, 3>,
    (void*)mx4::build_xfrag_p, (void*)mx4::k_fc1<5120, 576, 8, 1, 2, false, 1>, (void*)mx4::k_fc2<576, 5120, 8, 2, false, 1>,
    (void*)dsh::k_sh_fc1<5120, 576, 8, 1, 2, 1>, (void*)dsh::k_sh_fc2<576, 5120, 8, 3, 1>,
    (void*)dsr::k_gate_logits_mt<384, 1>, (void*)dsr::k_plan_mt<384, 6, 32, 1>, (void*)mx4::build_xfrag_mt<1>,
    (void*)mx4::k_fc1_mt<5120, 576, 8, 1, mx4::FC1_MT_DEPTH<2>, 2, 0, 1>, (void*)mx4::k_fc1_mt<5120, 576, 8, 1, mx4::FC1_MT_DEPTH<4>, 4, 0, 1>,
    (void*)mx4::k_fc2_mt<576, 5120, 8, 2, 2, 1>, (void*)mx4::k_fc2_mt<576, 5120, 8, 2, 4, 1>,
    (void*)dsh::k_sh_fc1_mt<5120, 576, 8, 1, 2, 2, 1>, (void*)dsh::k_sh_fc1_mt<5120, 576, 8, 1, 2, 4, 1>,
    (void*)dsh::k_sh_fc2_mt<576, 5120, 8, 3, 2, 1>, (void*)dsh::k_sh_fc2_mt<576, 5120, 8, 3, 4, 1>,
};
#endif
int main() { return g_ab_kernels[0] == nullptr; }
