module mod_solver
    use mod_streams

    !
     real(mykind) :: elapsed,startTiming,endTiming
    !
     integer :: i
     logical :: stop_streams
end module

! init_solver contains all the setup routines that should take place as part of 
subroutine init_solver
    use mod_streams
    use mod_solver
    implicit none

     icyc   = ncyc0
     telaps = telaps0
    !
    !Copy arrays from CPU to GPU or move alloc
     call copy_cpu_to_gpu()

    !
     if (masterproc) write(*,*) 'Compute time step'
     dtmin = abs(cfl)
     if (cfl>0._mykind) call step()
     if (masterproc) write(*,*) 'Done'
    !
     if (xrecyc>0._mykind) call recyc
     call updateghost() ! Needed here only for subsequent prims call
     call prims()
     if (tresduc>0._mykind.and.tresduc<1._mykind) then
      call sensor()
      call bcswapduc_prepare()
      call bcswapduc()
     endif
    !
     open(20,file='output_streams.dat',position=stat_io)

     startTiming = mpi_wtime()
    !
     stop_streams = .false.
end subroutine

subroutine step_solver()
    use mod_solver
    implicit none

  icyc = icyc+1
  itav = icyc
!
  call rk() ! Third-order RK scheme
 ! write the probe dtata before any reset occurs

!
  if (io_type>0) call manage_solver()
  !exit ! break after we have written the data for the first step - TODO: REMOVE ME
!
  if (mod(i,nprint)==0) then
   call computeresidual()
   call printres()
  endif
!
  if (cfl>0._mykind) then
   if (mod(i,nstep)==0) call step() ! Compute the time step
  endif
!
  call mpi_barrier(mpi_comm_world,iermpi)
!
end subroutine

subroutine finalize_solver()
    use mod_solver
    implicit none
!
 endTiming = mpi_wtime()
 elapsed = endTiming-startTiming
 if (ncyc>0) then
  if (masterproc) write(error_unit,*) 'Time-step time =', elapsed/ncyc
  if (masterproc) write(20,*) 'Time-step time =', elapsed/ncyc
 endif
!
end subroutine 

subroutine solver
!
! Solve the compressible NS equations
!
 use mod_streams
 use mod_solver
 implicit none
!

    ! initialize some solver stuff
    call init_solver()

 do i=1,ncyc
     call step_solver()
      inquire(file="stop.stop",exist=stop_streams)
      call mpi_barrier(mpi_comm_world,iermpi)
      if (stop_streams) exit
 enddo

 call finalize_solver()
end subroutine solver
