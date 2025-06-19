subroutine setup
!
! Setup for the computation
!
 use mod_streams
 implicit none
! find out the dimensions of the blowing slot
!
!===================================================
 if (masterproc) write(error_unit,*) 'Allocation of variables'
 call allocate_vars()
!===================================================
 if (masterproc) write(error_unit,*) 'Reading input'
 call readinp()

! separate allocations after the input files have been read so that 
! the correct boundary conditions may be established
 call local_slot_locations()
 call allocate_blowing_bcs()

!===================================================
 if (idiski==0) then
  if (masterproc) write(*,*) 'Generating mesh'
  call generategrid()
 else
  if (masterproc) write(*,*) 'Reading mesh'
  call readgrid()
 endif
 if (masterproc) write(*,*) 'Computing metrics'
 call computemetrics()
 if (enable_plot3d>0) call writegridplot3d()
!===================================================
 call constants()
!===================================================
 if (xrecyc>0._mykind) call recyc_prepare()
!===================================================
 if (masterproc) write(*,*) 'Initialize field'
 call init()
 call checkdt()
 call generate_full_fdm_stencil()
!===================================================
!
end subroutine setup
