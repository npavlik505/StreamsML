#! /bin/python3

# CPU array allocations. These arrays are /always/ allocated.
cpu_alloc = [
    "wv_gpu",
    "w_order",
    #
    "wallpfield_gpu",
    "slicexy_gpu",
    "vf_df_old",
    "uf",
    "evmax_mat_yz",
    "evmax_mat_y",
    "bulk5g_gpu",
    "rtrms_ib_gpu",
    "rtrms_ib_1d_gpu",
    #
    "wbuf1s_gpu",
    "wbuf2s_gpu",
    "wbuf3s_gpu",
    "wbuf4s_gpu",
    "wbuf5s_gpu",
    "wbuf6s_gpu",
    "wbuf1r_gpu",
    "wbuf2r_gpu",
    "wbuf3r_gpu",
    "wbuf4r_gpu",
    "wbuf5r_gpu",
    "wbuf6r_gpu",
    "divbuf1s_gpu",
    "divbuf2s_gpu",
    "divbuf3s_gpu",
    "divbuf4s_gpu",
    "divbuf5s_gpu",
    "divbuf6s_gpu",
    "divbuf1r_gpu",
    "divbuf2r_gpu",
    "divbuf3r_gpu",
    "divbuf4r_gpu",
    "divbuf5r_gpu",
    "divbuf6r_gpu",
    "ducbuf1s_gpu",
    "ducbuf2s_gpu",
    "ducbuf3s_gpu",
    "ducbuf4s_gpu",
    "ducbuf5s_gpu",
    "ducbuf6s_gpu",
    "ducbuf1r_gpu",
    "ducbuf2r_gpu",
    "ducbuf3r_gpu",
    "ducbuf4r_gpu",
    "ducbuf5r_gpu",
    "ducbuf6r_gpu",
    #
    "w",
    "fl",
    "fln",
    "temperature",
    "ducros",
    "wmean",
    "dcsidx",
    "dcsidx2",
    "dcsidxs",
    "detady",
    "detady2",
    "detadys",
    "dzitdz",
    "dzitdz2",
    "dzitdzs",
    "dcsidxh",
    "detadyh",
    "dzitdzh",
    "x",
    "y",
    "yn",
    "yn_gpu",
    "z",
    "xg",
    "coeff_deriv1",
    "coeff_deriv1s",
    "coeff_clap",
    "coeff_midpi",
    "cx_midpi",
    "cy_midpi",
    "cz_midpi",
    "fhat",
    "ibcnr",
    "dcoe",
    "winf",
    "winf1",
    "rf",
    "rfy",
    "vf_df",
    "by_df",
    "bz_df",
    "amat_df",
    "wallpfield",
    "slicexy",
    "xh",
    "yh",
    "zh",
    "xgh",
    "ygh",
    "zgh",
    "yplus_inflow",
    "yplus_recyc",
    "eta_inflow",
    "eta_recyc",
    "map_j_inn",
    "map_j_out",
    "weta_inflow",
    #
    "ibc",
    "dxg",
    "dyg",
    "dzg",
    "w_av",
    "w_avzg",
    "w_av_1d",
    "w_avxzg",
    "bx_df",
    "wbuf1s",
    "wbuf2s",
    "wbuf3s",
    "wbuf4s",
    "wbuf5s",
    "wbuf6s",
    "wbuf1r",
    "wbuf2r",
    "wbuf3r",
    "wbuf4r",
    "wbuf5r",
    "wbuf6r",
    "divbuf1s",
    "divbuf2s",
    "divbuf3s",
    "divbuf4s",
    "divbuf5s",
    "divbuf6s",
    "divbuf1r",
    "divbuf2r",
    "divbuf3r",
    "divbuf4r",
    "divbuf5r",
    "divbuf6r",
    "ducbuf1s",
    "ducbuf2s",
    "ducbuf3s",
    "ducbuf4s",
    "ducbuf5s",
    "ducbuf6s",
    "ducbuf1r",
    "ducbuf2r",
    "ducbuf3r",
    "ducbuf4r",
    "ducbuf5r",
    "ducbuf6r",
    "yg",
    "zg",
    #
    "wrecyc_gpu",
    "wrecycav_gpu",
    #
    "tauw_x",
    #
    "fdm_y_stencil_gpu",
    "fdm_y_stencil",
    "fdm_individual_stencil",
    "fdm_grid_points",
]

# statements that are located in a #IFDEF CUDA
# block. These allocations are /always/ made, as long as cuda is enabled
ifdef_cuda = [
    "fl_trans_gpu",
    "temperature_trans_gpu",
    "fhat_trans_gpu",
    #
    "wv_trans_gpu",
    #
    "w_gpu",
    "fl_gpu",
    "fln_gpu",
    "temperature_gpu",
    "ducros_gpu",
    "wmean_gpu",
    "dcsidx_gpu",
    "dcsidx2_gpu",
    "dcsidxs_gpu",
    "detady_gpu",
    "detady2_gpu",
    "detadys_gpu",
    "dzitdz_gpu",
    "dzitdz2_gpu",
    "dzitdzs_gpu",
    "dcsidxh_gpu",
    "detadyh_gpu",
    "dzitdzh_gpu",
    "x_gpu",
    "y_gpu",
    "z_gpu",
    "xg_gpu",
    "coeff_deriv1_gpu",
    "coeff_deriv1s_gpu",
    "coeff_clap_gpu",
    "coeff_midpi_gpu",
    "cx_midpi_gpu",
    "cy_midpi_gpu",
    "cz_midpi_gpu",
    "fhat_gpu",
    "ibcnr_gpu",
    "dcoe_gpu",
    "winf_gpu",
    "rf_gpu",
    "rfy_gpu",
    "vf_df_gpu",
    "by_df_gpu",
    "bz_df_gpu",
    "amat_df_gpu",
    #
    "yplus_inflow_gpu",
    "yplus_recyc_gpu",
    "eta_inflow_gpu",
    "eta_recyc_gpu",
    "map_j_inn_gpu",
    "map_j_out_gpu",
    "weta_inflow_gpu",
    #
    "gplus_x",
    "gminus_x",
    "gplus_y",
    "gminus_y",
    "gplus_z",
    "gminus_z",
    #
]


# these are allocations that are not present in alloc.f90. They will be 
# deallocated behind if statements to ensure that we dont deallocate an array
# that is not allocated.
#
# note that it is possible that these arrays are /always/ allocated, but the runtime
# penalty for the deallocation is minimal
optional_allocations = [
    # from readinp.f90
    "xstat",
    "ixstat",
    "igxstat",
    # from bcblow.f90
    "blowing_bc_slot_velocity",
    "blowing_bc_slot_velocity_gpu",
    # startmpi.f90
    "ncoords",
    "nblocks",
    "pbc",
]

def deallocate(var):
    return f"deallocate({var})"

def deallocate_if_allocated(var):
    return f"if (allocated({var})) deallocate({var})"

code = "subroutine deallocate_all() \nuse mod_streams\n\n"

for array in cpu_alloc:
    code += deallocate(array)
    code += "\n"

code += "#ifdef USE_CUDA\n"

for array in ifdef_cuda:
    code += deallocate(array)
    code += "\n"

code += "#endif\n"

for array in optional_allocations:
    code += deallocate_if_allocated(array)
    code += "\n"

code +="end subroutine deallocate_all"


print(code)
