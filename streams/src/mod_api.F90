module mod_api_work_buffers
    use mod_streams, only: mykind
    implicit none
    real(mykind), allocatable, save :: tauw_host(:)
    real(mykind), allocatable, save :: w_avzg_host(:,:)
contains
    subroutine ensure_tauw_host_size(required_len)
        integer, intent(in) :: required_len

        if (required_len <= 0) then
            if (allocated(tauw_host)) then
                deallocate(tauw_host)
            end if
            return
        end if

        if (.not. allocated(tauw_host)) then
            allocate(tauw_host(required_len))
        else if (size(tauw_host) /= required_len) then
            deallocate(tauw_host)
            allocate(tauw_host(required_len))
        end if
    end subroutine ensure_tauw_host_size

    subroutine ensure_w_avzg_host_size(nx_slice, ny_slice)
        integer, intent(in) :: nx_slice, ny_slice

        if (nx_slice <= 0 .or. ny_slice <= 0) then
            if (allocated(w_avzg_host)) then
                deallocate(w_avzg_host)
            end if
            return
        end if

        if (.not. allocated(w_avzg_host)) then
            allocate(w_avzg_host(nx_slice, ny_slice))
        else if (size(w_avzg_host, 1) /= nx_slice .or. size(w_avzg_host, 2) /= ny_slice) then
            deallocate(w_avzg_host)
            allocate(w_avzg_host(nx_slice, ny_slice))
        end if
    end subroutine ensure_w_avzg_host_size
end module mod_api_work_buffers

!---- Data Retrieving Subroutines ----

subroutine wrap_get_x(x_out, i_start, i_end) bind(C, name="wrap_get_x")
    use iso_c_binding, only: c_int, c_float
    use mod_streams,      only: x
    !f2py intent(c) wrap_get_x
    !f2py intent(out) x_out
    implicit none
    integer(c_int), intent(in), value :: i_start, i_end
    integer                         :: len, i
    real(c_float), intent(out)      :: x_out(i_end - i_start)
    len = i_end - i_start
    do i = 1, len
      x_out(i) = real(x(i_start + i - 1), kind=c_float)
    end do
end subroutine wrap_get_x

subroutine wrap_get_y(y_out, i_start, i_end) bind(C, name="wrap_get_y")
    use iso_c_binding, only: c_int, c_float
    use mod_streams,      only: y
    !f2py intent(c) wrap_get_y
    !f2py intent(out) y_out
    implicit none
    integer(c_int), intent(in), value :: i_start, i_end
    integer                         :: len, i
    real(c_float), intent(out)      :: y_out(i_end - i_start)
    len = i_end - i_start
    do i = 1, len
      y_out(i) = real(y(i_start + i - 1), kind=c_float)
    end do
end subroutine wrap_get_y

subroutine wrap_get_z(z_out, i_start, i_end) bind(C, name="wrap_get_z")
    use iso_c_binding, only: c_int, c_float
    use mod_streams,      only: z
    !f2py intent(c) wrap_get_z
    !f2py intent(out) z_out
    implicit none
    integer(c_int), intent(in), value :: i_start, i_end
    integer                         :: len, i
    real(c_float), intent(out)      :: z_out(i_end - i_start)
    len = i_end - i_start
    do i = 1, len
      z_out(i) = real(z(i_start + i - 1), kind=c_float)
    end do
end subroutine wrap_get_z

subroutine wrap_get_tauw_x(tauw_out, n) bind(C, name="wrap_get_tauw_x")
    use iso_c_binding, only: c_double
    use mod_streams, only: tauw_x, tauw_x_gpu, mykind
    use mod_api_work_buffers, only: tauw_host, ensure_tauw_host_size
#ifdef USE_CUDA
    use cudafor
#endif
    !f2py intent(c) wrap_get_tauw_x
    !f2py intent(out) tauw_out
    implicit none
    integer, intent(in), value :: n
    real(c_double), intent(out) :: tauw_out(n)
    integer :: ncopy

    if (n <= 0) then
        return
    end if
    if (.not. allocated(tauw_x)) then
        tauw_out = 0.0_c_double
        return
    end if
    ncopy = min(n, size(tauw_x))
    if (ncopy <= 0) then
        tauw_out = 0.0_c_double
        return
    end if
#ifdef USE_CUDA
    if (allocated(tauw_x_gpu)) then
        call ensure_tauw_host_size(ncopy)
        tauw_host(1:ncopy) = tauw_x_gpu(1:ncopy)
        tauw_out(1:ncopy) = real(tauw_host(1:ncopy), kind=c_double)
        if (ncopy < n) tauw_out(ncopy + 1:n) = 0.0_c_double
        return
    end if
#endif
    tauw_out(1:ncopy) = real(tauw_x(1:ncopy), kind=c_double)
    if (ncopy < n) tauw_out(ncopy + 1:n) = 0.0_c_double
end subroutine wrap_get_tauw_x

subroutine wrap_get_w(w_out, n1, n2, n3, n4) bind(C, name="wrap_get_w")
    use iso_c_binding, only: c_int, c_float
    use mod_streams,      only: w
    !f2py intent(c) wrap_get_w
    !f2py intent(out) w_out
    implicit none
    integer(c_int), intent(in), value :: n1, n2, n3, n4
    real   (c_float), intent(out)      :: w_out(n1, n2, n3, n4)
    w_out = real(w, kind=c_float)
end subroutine wrap_get_w

subroutine wrap_get_w_avzg_slice(w_avzg_out, obs_xstart, obs_xend, obs_ystart, obs_yend, obs_type) bind(C, name="wrap_get_w_avzg_slice")
    use iso_c_binding, only: c_int, c_float
    use mod_streams,      only: w_avzg
    !f2py intent(c) wrap_get_w_avzg_slice
    !f2py intent(out) w_avzg_out
    implicit none
    integer(c_int), intent(in), value :: obs_xstart, obs_xend, obs_ystart, obs_yend, obs_type
    real(c_float), intent(out) :: w_avzg_out(1, obs_xend - obs_xstart + 1, obs_yend - obs_ystart + 1)
    w_avzg_out(1,:,:) = real(w_avzg(obs_type, obs_xstart:obs_xend, obs_ystart:obs_yend), kind=c_float)
end subroutine wrap_get_w_avzg_slice

subroutine wrap_get_w_avzg_slice_gpu(w_avzg_out, obs_xstart, obs_xend, obs_ystart, obs_yend, obs_type) bind(C, name="wrap_get_w_avzg_slice_gpu")
    use iso_c_binding, only: c_int, c_float
    use mod_streams,      only: w_avzg, w_avzg_gpu, mykind
    use mod_api_work_buffers, only: w_avzg_host, ensure_w_avzg_host_size
#ifdef USE_CUDA
    use cudafor
#endif
    !f2py intent(c) wrap_get_w_avzg_slice_gpu
    !f2py intent(out) w_avzg_out
    implicit none
    integer(c_int), intent(in), value :: obs_xstart, obs_xend, obs_ystart, obs_yend, obs_type
    integer :: nx_slice, ny_slice
    real(c_float), intent(out) :: w_avzg_out(1, obs_xend - obs_xstart + 1, obs_yend - obs_ystart + 1)

    nx_slice = obs_xend - obs_xstart + 1
    ny_slice = obs_yend - obs_ystart + 1
    if (nx_slice <= 0 .or. ny_slice <= 0) then
        w_avzg_out = 0.0_c_float
        return
    end if
    if (.not. allocated(w_avzg)) then
        w_avzg_out = 0.0_c_float
        return
    end if
    if (obs_type < 1 .or. obs_type > size(w_avzg, 1)) then
        w_avzg_out = 0.0_c_float
        return
    end if
#ifdef USE_CUDA
    if (allocated(w_avzg_gpu)) then
        call ensure_w_avzg_host_size(nx_slice, ny_slice)
        w_avzg_host(:, :) = w_avzg_gpu(obs_type, obs_xstart:obs_xend, obs_ystart:obs_yend)
        w_avzg_out(1,:,:) = real(w_avzg_host(:, :), kind=c_float)
        return
    end if
#endif
    w_avzg_out(1,:,:) = real(w_avzg(obs_type, obs_xstart:obs_xend, obs_ystart:obs_yend), kind=c_float)
end subroutine wrap_get_w_avzg_slice_gpu

subroutine wrap_get_x_start_slot(val) bind(C, name="wrap_get_x_start_slot")
    use iso_c_binding
    use mod_streams, only: x_start_slot
    !f2py intent(c) wrap_get_x_start_slot
    !f2py intent(out) val
    implicit none
    integer, intent(out) :: val
    val = x_start_slot
end subroutine wrap_get_x_start_slot

subroutine wrap_get_nx_slot(val) bind(C, name="wrap_get_nx_slot")
    use iso_c_binding
    use mod_streams, only: nx_slot
    !f2py intent(c) wrap_get_nx_slot
    !f2py intent(out) val
    implicit none
    integer, intent(out) :: val
    ! print *, '>>> wrap_get_nx_slot called, nx_slot =', nx_slot
    val = nx_slot
end subroutine wrap_get_nx_slot

subroutine wrap_get_nz_slot(val) bind(C, name="wrap_get_nz_slot")
    use iso_c_binding
    use mod_streams, only: nz_slot
    !f2py intent(c) wrap_get_nz_slot
    !f2py intent(out) val
    implicit none
    integer, intent(out) :: val
    val = nz_slot
end subroutine wrap_get_nz_slot

subroutine wrap_get_blowing_bc_slot_velocity(arr, n1, n2) bind(C, name="wrap_get_blowing_bc_slot_velocity")
    use iso_c_binding, only: c_int, c_float
    use mod_streams,      only: blowing_bc_slot_velocity
    !f2py intent(c) wrap_get_blowing_bc_slot_velocity
    !f2py intent(out) arr
    implicit none
    integer(c_int), intent(in), value :: n1, n2
    real   (c_float), intent(out)      :: arr(n1, n2)
    arr = real(blowing_bc_slot_velocity, kind=c_float)
end subroutine wrap_get_blowing_bc_slot_velocity

subroutine wrap_set_nx_slot(val) bind(C, name="wrap_set_nx_slot")
    use iso_c_binding
    use mod_streams, only: nx_slot
    !f2py intent(c) wrap_set_nx_slot
    !f2py intent(in) val
    implicit none
    integer, intent(in) :: val
    nx_slot = val
    ! print *, '>>> wrap_set_nx_slot: setting nx_slot =', val
end subroutine wrap_set_nx_slot

subroutine wrap_get_dtglobal(dt_out) bind(C, name="wrap_get_dtglobal")
    use iso_c_binding
    use mod_streams, only: dtglobal
    !f2py intent(c) wrap_get_dtglobal
    !f2py intent(out) dt_out
    implicit none
    real(8), intent(out) :: dt_out
    dt_out = dtglobal
end subroutine wrap_get_dtglobal

subroutine wrap_get_dissipation_rate(val) bind(C, name="wrap_get_dissipation_rate")
  use iso_c_binding, only: c_float
  use mod_streams,      only: dissipation_rate
  !f2py intent(c) wrap_get_dissipation_rate
  !f2py intent(out) val
  implicit none
  real(c_float), intent(out) :: val
  val = real(dissipation_rate, kind=c_float)
end subroutine wrap_get_dissipation_rate

subroutine wrap_get_energy(val) bind(C, name="wrap_get_energy")
  use iso_c_binding, only: c_float
  use mod_streams,      only: energy
  !f2py intent(c) wrap_get_energy
  !f2py intent(out) val
  implicit none
  real(c_float), intent(out) :: val
  val = real(energy, kind=c_float)
end subroutine wrap_get_energy

!---- Shape Retrieving Subroutines ----

subroutine wrap_get_w_shape(n1, n2, n3, n4) bind(C, name="wrap_get_w_shape")
    use iso_c_binding, only: c_int
    use mod_streams,      only: w
    !f2py intent(c) wrap_get_w_shape
    !f2py intent(out) n1, n2, n3, n4
    implicit none
    integer(c_int), intent(out) :: n1, n2, n3, n4
    n1 = size(w, 1)
    n2 = size(w, 2)
    n3 = size(w, 3)
    n4 = size(w, 4)
end subroutine wrap_get_w_shape

subroutine wrap_get_blowing_bc_slot_velocity_shape(n1, n2) bind(C, name="wrap_get_blowing_bc_slot_velocity_shape")
    use iso_c_binding, only: c_int
    use mod_streams,      only: blowing_bc_slot_velocity
    !f2py intent(c) wrap_get_blowing_bc_slot_velocity_shape
    !f2py intent(out) n1, n2
    implicit none
    integer(c_int), intent(out) :: n1, n2
    n1 = size(blowing_bc_slot_velocity,1)
    n2 = size(blowing_bc_slot_velocity,2)
end subroutine wrap_get_blowing_bc_slot_velocity_shape

subroutine wrap_get_tauw_x_shape(n) bind(C, name="wrap_get_tauw_x_shape")
    use iso_c_binding, only: c_int
    use mod_streams,      only: tauw_x
    !f2py intent(c) wrap_get_tauw_x_shape
    !f2py intent(out) n
    implicit none
    integer(c_int), intent(out) :: n
    n = size(tauw_x)
end subroutine wrap_get_tauw_x_shape
