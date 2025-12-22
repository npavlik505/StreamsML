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
    use iso_c_binding
    use mod_streams, only: tauw_x
    !f2py intent(c) wrap_get_tauw_x
    !f2py intent(out) tauw_out
    implicit none
    integer, intent(in), value :: n
    real(8), intent(out) :: tauw_out(n)
    integer :: i

    do i = 1, n
        tauw_out(i) = tauw_x(i)
    end do
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

subroutine wrap_get_w_avzg_points(values_out, xs, ys, n_points, obs_type) bind(C, name="wrap_get_w_avzg_points")
    use iso_c_binding, only: c_int, c_float
    use mod_streams,      only: w_avzg
    !f2py intent(c) wrap_get_w_avzg_points
    !f2py intent(out) values_out
    implicit none
    integer(c_int), intent(in), value :: n_points, obs_type
    integer(c_int), intent(in) :: xs(n_points), ys(n_points)
    real(c_float), intent(out) :: values_out(n_points)
    integer :: i
    do i = 1, n_points
        values_out(i) = real(w_avzg(obs_type, xs(i), ys(i)), kind=c_float)
    end do
end subroutine wrap_get_w_avzg_points

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

subroutine wrap_set_blowing_bc_slot_velocity(arr, n1, n2) bind(C, name="wrap_set_blowing_bc_slot_velocity")
    use iso_c_binding, only: c_int, c_float
    use mod_streams,      only: blowing_bc_slot_velocity, mykind
    !f2py intent(c) wrap_set_blowing_bc_slot_velocity
    !f2py intent(inout) arr
    implicit none
    integer(c_int), intent(in), value :: n1, n2
    real   (c_float), intent(inout)      :: arr(n1, n2)

    blowing_bc_slot_velocity(1:n1, 1:n2) = real(arr, kind=mykind)
    arr = real(blowing_bc_slot_velocity(1:n1, 1:n2), kind=c_float)
end subroutine wrap_set_blowing_bc_slot_velocity

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
