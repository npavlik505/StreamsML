#----------------------------------------------------------------------
# SUPPORTED COMPILE: pgi-cuda, pgi-cuda-sync, pgi, intel, gnu, ibmxl, cray-cuda
# SUPPORTED MODES  : opt, debug [default=debug]
# SUPPORTED PRECISIONS (PREC) : single, double [default=double]
#----------------------------------------------------------------------
# by default, compile with the GNU compiler
# in the singularity build script for GPU mode this will be automatically
# replaced with 
# COMPILE = "nvfortran"
# caveat: the string below must be EXACTLY: `COMPILE = "gnu"`
COMPILE = "gnu"
MODE    = "debug"
PREC    = "double"
#----------------------------------------------------------------------
ifeq (${PREC},"single") # mpif90 shipped with PGI compiler
    PRECFLAGS = -DSINGLE_PRECISION 
endif
#----------------------------------------------------------------------

# singularity build nvhpc_21.1_devel.sif docker://nvcr.io/nvidia/nvhpc:21.1-devel-cuda_multi-ubuntu20.04
# singularity shell --nv nvhpc_21.1_devel.sif
ifeq (${COMPILE},"nvfortran") # mpif90 shipped with PGI compiler
    FC = mpifort

    ifeq (${MODE},"opt") 
		# everything after async is bonus, not sure if required
        OLEVEL = -O3 -cuda -gpu=ccall -DUSE_CUDA -DCUDA_ASYNC
    else
        OLEVEL = -cuda -gpu=ccall -DUSE_CUDA -DCUDA_ASYNC -g -DNOCUDAAWAREMPI
    endif
endif

ifeq (${COMPILE},"gnu") # mpif90 from MPI library
    FC = mpifort
    ifeq (${MODE},"opt") 
        OLEVEL = -O3 #-fno-inline-functions -fno-underscoring -ffree-line-length-512
    else
        OLEVEL = -O2 -fbounds-check -g -fbacktrace -fdump-core -ffpe-trap=zero,invalid,overflow -Wuninitialized -Wall -ffree-line-length-512
    endif
endif

RM = rm -f

#----------------------------------------------------------------------
# Base code
#----------------------------------------------------------------------
MAIN = main.o
OBJ_FILES = alloc.o bcdf.o bcextr.o bcfree.o bc.o bcrecyc.o bcrelax.o bcshk.o bcswap.o \
    bcwall.o bcwall_pl.o bcwall_staggered.o check_input.o computeav.o computemetrics.o \
    computeresidual.o compute_coeff_xyz_midpi.o constants.o df_par.o digital_filtering_util.o \
    finalize.o generategrid.o generateinflowrand.o generatewmean_channel.o \
    generatewmean.o heatflux_compute.o init_channel.o init.o initurb.o init_windtunnel.o manage_solver.o osw.o pgrad.o \
    prims.o printres.o readdf.o readgrid.o readinp.o readrst.o readstat.o recyc.o rk.o sensor.o \
    setup.o solver.o startmpi.o stats.o step.o target_reystress.o tbforce.o updateghost.o utility.o utyibm.o \
    visflx.o writedf.o writefield.o writefieldvtk.o writegridplot3d.o write_probe_data.o writerst.o \
    writestatbl.o writestatchann.o writestat.o writestatzbl.o write_wallpressure.o visflx_stag.o \
	bcblow.o dissipation.o dealloc.o

PY_API_OBJS = min_api.o mod_api.o

STATIC_FILES = alloc.F90 bcdf.F90 bcextr.F90 bcfree.F90 bc.F90 bcrecyc.F90 bcrelax.F90 bcshk.F90 bcswap.F90 \
    bcwall.F90 bcwall_pl.F90 bcwall_staggered.F90 check_input.F90 computeav.F90 computemetrics.F90 \
    computeresidual.F90 compute_coeff_xyz_midpi.F90 constants.F90 df_par.F90 digital_filtering_util.F90 \
    finalize.F90 generategrid.F90 generateinflowrand.F90 generatewmean_channel.F90 \
    generatewmean.F90 heatflux_compute.F90 init_channel.F90 init.F90 initurb.F90 init_windtunnel.F90 manage_solver.F90 osw.F90 pgrad.F90 \
    prims.F90 printres.F90 readdf.F90 readgrid.F90 readinp.F90 readrst.F90 readstat.F90 recyc.F90 rk.F90 sensor.F90 \
    setup.F90 solver.F90 startmpi.F90 stats.F90 step.F90 target_reystress.F90 tbforce.F90 updateghost.F90 utility.F90 utyibm.F90 \
    visflx.F90 writedf.F90 writefield.F90 writefieldvtk.F90 writegridplot3d.F90 write_probe_data.F90 writerst.F90 \
    writestatbl.F90 writestatchann.F90 writestat.F90 writestatzbl.F90 write_wallpressure.F90 visflx_stag.F90 \
	bcblow.F90 dissipation.F90 dealloc.F90

STATIC_MODS = mod_streams.F90 mod_sys.F90 euler.F90

STATIC_MAIN = main.F90

OBJ_MODS = mod_streams.o mod_sys.o euler.o
OBJS = $(OBJ_MODS) $(OBJ_FILES) $(PY_API_OBJS) $(MAIN)
EXEC = streams.exe
#LDFLAGS = 
SHARED = -fPIC

# Creating streams.exe
$(EXEC): $(OBJS) Makefile $(wildcard makefile.inc)
	$(FC) $(PRECFLAGS) $(OLEVEL) -o $@ $(OBJS) $(FCLIBS) $(LDFLAGS)

# Ensure min_api and mod_api use the same mod_streams mod
mod_streams.o: mod_streams.F90
	 $(FC) $(PRECFLAGS) $(OLEVEL) $(SHARED) -c $< -o $@

mod_streams.mod: mod_streams.o

mod_api.o min_api.o: mod_streams.mod

# Rules for converting .F90, .cpp, and .cxx files into object files
%.o:%.F90 Makefile $(wildcard makefile.inc)
	$(FC) $(PRECFLAGS) $(INCLUDE) $(OLEVEL) $(SHARED) -c $< -o $@

%.o:%.cpp Makefile $(wildcard makefile.inc)
	$(CXX) $(PRECFLAGS) $(INCLUDE) -O2 -c $< -o $@

%.o:%.cxx Makefile $(wildcard makefile.inc)
	$(CXX) $(PRECFLAGS) $(INCLUDE) -O2 -c $< -o $@

# f2py Wrapping routine (runs at "make f2py" in build.apptainer) 
static:
	mpifort $(STATIC_MODS) $(STATIC_FILES) $(STATIC_MAIN) -o $(EXEC)

libstreams.pyf:
	f2py -m libstreams -h libstreams.pyf --overwrite-signature min_api.F90 mod_api.F90

patched_pyfs: libstreams.pyf
	python3 /streams-utils/patch_pyf.py

f2py: patched_pyfs $(PY_API_OBJS) $(OBJ_FILES) $(OBJ_MODS)
	F2PY_FCOMPILER=nv \
	F90=$(shell which mpifort) \
	NPY_DISTUTILS_APPEND_FLAGS=1 \
	FFLAGS="$(FFLAGS)" LDFLAGS="$(LDFLAGS)" \
	f2py --fcompiler=nv -c libstreams.pyf \
	     $(PY_API_OBJS) $(OBJ_FILES) $(OBJ_MODS) \
	     -m libstreams

# Files to delete when running "make clean"
.PHONY: clean
clean:
	$(RM) $(EXEC) $(MAIN) $(OBJS) $(OBJ_MODS) $(GRID) *.mod *.so *.pyf *.pyf.raw cfz_*.dat *.vtr *.q *.csv *.prof *.bin cf_*.dat d*g.dat *.xyz wavplot_*.dat csv_data/* spans/* *.vtk plotxy*.dat x.dat y.dat z.dat mod_streams_pre.F90 mod_streams.pyf
