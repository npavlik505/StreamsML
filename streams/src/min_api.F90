subroutine wrap_startmpi() bind(C, name="wrap_startmpi")
    use iso_c_binding
    !f2py intent(c) wrap_startmpi
    !f2py intent(hide)
    call startmpi()
end subroutine wrap_startmpi

subroutine wrap_setup() bind(C, name="wrap_setup")
    use iso_c_binding
    !f2py intent(c) wrap_setup
    !f2py intent(hide)
    use mod_streams
    call setup()
end subroutine wrap_setup

subroutine wrap_init_solver() bind(C, name="wrap_init_solver")
    use iso_c_binding
    !f2py intent(c) wrap_init_solver
    !f2py intent(hide)
    call init_solver()
end subroutine wrap_init_solver

subroutine wrap_step_solver() bind(C, name="wrap_step_solver")
    use iso_c_binding
    !f2py intent(c) wrap_step_solver
    !f2py intent(hide)
    call step_solver()
end subroutine wrap_step_solver

subroutine wrap_finalize_solver() bind(C, name="wrap_finalize_solver")
    use iso_c_binding
    !f2py intent(c) wrap_finalize_solver
    !f2py intent(hide)
    call finalize_solver()
end subroutine wrap_finalize_solver

subroutine wrap_finalize() bind(C, name="wrap_finalize")
    use iso_c_binding
    !f2py intent(c) wrap_finalize
    !f2py intent(hide)
    use mod_streams
    call finalize()
    call mpi_finalize(iermpi)
end subroutine wrap_finalize

subroutine wrap_copy_gpu_to_cpu() bind(C, name="wrap_copy_gpu_to_cpu")
    use iso_c_binding
    !f2py intent(c) wrap_copy_gpu_to_cpu
    !f2py intent(hide)
    call updateghost()
    call prims()
    call copy_gpu_to_cpu()
end subroutine wrap_copy_gpu_to_cpu

subroutine wrap_copy_cpu_to_gpu() bind(C, name="wrap_copy_cpu_to_gpu")
    use iso_c_binding
    !f2py intent(c) wrap_copy_cpu_to_gpu
    !f2py intent(hide)
    call copy_cpu_to_gpu()
end subroutine wrap_copy_cpu_to_gpu

subroutine wrap_copy_blowing_bc_to_gpu() bind(C, name="wrap_copy_blowing_bc_to_gpu")
    use iso_c_binding
    !f2py intent(c) wrap_copy_blowing_bc_to_gpu
    !f2py intent(hide)
    call copy_blowing_bc_to_gpu()
end subroutine wrap_copy_blowing_bc_to_gpu

subroutine wrap_copy_blowing_bc_to_cpu() bind(C, name="wrap_copy_blowing_bc_to_cpu")
    use iso_c_binding
    !f2py intent(c) wrap_copy_blowing_bc_to_cpu
    !f2py intent(hide)
    call copy_blowing_bc_to_cpu()
end subroutine wrap_copy_blowing_bc_to_cpu

subroutine wrap_tauw_calculate(averages_current) bind(C, name="wrap_tauw_calculate")
    use iso_c_binding
    !f2py intent(c) wrap_tauw_calculate
    !f2py intent(in) averages_current
    use mod_streams, only: tauw_x, tauw_x_gpu, w_avzg, w_avzg_gpu, mykind, y, y_gpu, nx, ny, ncoords
    implicit none
    integer :: i, j
    integer(c_int), value :: averages_current
    logical :: averages_current_logical
#ifdef USE_CUDA
    real(mykind) :: uf1, uf2, uf3, uf4
    real(mykind) :: dudyw_local, dy_local, rmuw_local
#endif

    averages_current_logical = (averages_current /= 0)
    if (.not. averages_current_logical) then
        call compute_av()
    end if
#ifdef USE_CUDA
    if (ncoords(3) == 0) then
        !$cuf kernel do(1) <<<*,*>>>
        do i = 1, nx
            uf1 = w_avzg_gpu(13, i, 1)/w_avzg_gpu(1, i, 1)
            uf2 = w_avzg_gpu(13, i, 2)/w_avzg_gpu(1, i, 2)
            uf3 = w_avzg_gpu(13, i, 3)/w_avzg_gpu(1, i, 3)
            uf4 = w_avzg_gpu(13, i, 4)/w_avzg_gpu(1, i, 4)
            dudyw_local = (-22._mykind*uf1 + 36._mykind*uf2 - 18._mykind*uf3 + 4._mykind*uf4)/12._mykind
            dy_local = (-22._mykind*y_gpu(1) + 36._mykind*y_gpu(2) - 18._mykind*y_gpu(3) + 4._mykind*y_gpu(4))/12._mykind
            dudyw_local = dudyw_local/dy_local
            rmuw_local = w_avzg_gpu(20, i, 1)
            tauw_x_gpu(i) = rmuw_local*dudyw_local
        end do
        tauw_x = tauw_x_gpu
    end if
#else
    real(mykind), dimension(nx, ny) :: ufav
    real(mykind) :: dudyw, dy, rmuw, tauw

    if (ncoords(3) == 0) then
        do j = 1, ny
            do i = 1, nx
                ufav(i, j) = w_avzg(13, i, j)/w_avzg(1, i, j)
            end do
        end do
        do i = 1, nx
            dudyw = (-22._mykind*ufav(i, 1) + 36._mykind*ufav(i, 2) - 18._mykind*ufav(i, 3) + 4._mykind*ufav(i, 4))/12._mykind
            dy = (-22._mykind*y(1) + 36._mykind*y(2) - 18._mykind*y(3) + 4._mykind*y(4))/12._mykind
            dudyw = dudyw/dy
            rmuw = w_avzg(20, i, 1)
            tauw = rmuw*dudyw
            tauw_x(i) = tauw
        end do
    end if
#endif
#ifdef USE_CUDA
    if (allocated(tauw_x_gpu)) then
        tauw_x_gpu = tauw_x
    end if
#endif
end subroutine wrap_tauw_calculate

subroutine wrap_compute_av() bind(C, name="wrap_compute_av")
    use iso_c_binding
    use mod_streams
    !f2py intent(c) wrap_compute_av
    !f2py intent(hide)
    call updateghost()
    call prims()
    call compute_av()
end subroutine wrap_compute_av

subroutine wrap_dissipation_calculation() bind(C, name="wrap_dissipation_calculation")
    use iso_c_binding
    !f2py intent(c) wrap_dissipation_calculation
    !f2py intent(hide)
    implicit none
    call dissipation_calculation()
end subroutine wrap_dissipation_calculation

subroutine wrap_energy_calculation() bind(C, name="wrap_energy_calculation")
    use iso_c_binding
    !f2py intent(c) wrap_energy_calculation
    !f2py intent(hide)
    implicit none
    call energy_calculation()
end subroutine wrap_energy_calculation

subroutine wrap_deallocate_all() bind(C, name="wrap_deallocate_all")
    use iso_c_binding
    !f2py intent(c) wrap_deallocate_all
    !f2py intent(hide)
    implicit none
    call deallocate_all()
end subroutine wrap_deallocate_all
