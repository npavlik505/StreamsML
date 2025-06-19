! blowing boundary condition
!
! this may ONLY be called if we are on the bottom surface
! ilat == 3
! and have properly allocated the correct variables (blowing_bc_slot_velocity, blowing_bc_slot_velocity_gpu) in alloc.f90
subroutine bcblow(ilat)
    use mod_streams
    implicit none

    integer :: ilat
    integer :: i,j,k,l
    real(mykind) :: jet_velocity, rho
    real(mykind) :: uu,vv,ww,qq,pp,tt,rhoe

    j = 1

    ! if on the bottom
    if (ilat == 3) then
        ! call the the base boundary condition for the bottom wall first.
        ! I believe we dont use bcwall_pl here since it is categorized under
        ! the unsteady conditions. 
        ! Numerically, im not sure if that is the correct decision, so it
        ! may be something to better investigate in the future
        call bcwall(ilat)

        ! check if we should even worry about writing slot velocity information
        if (x_start_slot /= -1) then
            !write(*,*) "calling bcblow on bottom wall"

            ! w_gpu(:, :, :, 1) -> rho
            ! w_gpu(:, :, :, 2) -> rho u
            ! w_gpu(:, :, :, 3) -> rho v
            ! w_gpu(:, :, :, 4) -> rho w
            ! w_gpu(:, :, :, 5) -> rho E

            ! this -1 in the i indexing is counteracted by by the +1 on the 
            ! jet_velocity indexing, and this allows us to not
            ! use custom indexing on the array which makes interoperability with
            ! numpy easier

            !$cuf kernel do(2) <<<*,*>>>
            do i = x_start_slot,x_start_slot+nx_slot-1
                do k = 1,nz
#ifdef USE_CUDA
                    ! we have to shift around the indexing of the blowing BC 
                    ! array since `i` is indexing based on the entire local domain
                    ! but blowing_bc_slot_velocity is allocated with reference to only 
                    ! the size of the slot
                    jet_velocity = blowing_bc_slot_velocity_gpu(i-x_start_slot+1,k)


                    ! w() is indexed differently on CPU and GPU
                    rho = w_gpu(i,j,k,1)
                    w_gpu(i,j,k,3) = rho * jet_velocity

                    ! update the values at the ghost nodes. This is pretty similar to 
                    ! the ghost nodes of bcwall, but instead we edit the y-velocity a little differently
                    do l=1,ng
                        rho  = w_gpu(i,1+l,k,1)
                        uu   = w_gpu(i,1+l,k,2)/w_gpu(i,1+l,k,1)
                        vv   = w_gpu(i,1+l,k,3)/w_gpu(i,1+l,k,1)
                        ww   = w_gpu(i,1+l,k,4)/w_gpu(i,1+l,k,1)
                        rhoe = w_gpu(i,1+l,k,5)

                        qq   = 0.5_mykind*(uu*uu+vv*vv+ww*ww)
                        pp   = gm1*(rhoe-rho*qq)
                        tt   = pp/rho
                        tt   = 2._mykind*twall-tt
                        rho  = pp/tt

                        w_gpu(i,1-l,k,1) =  rho
                        w_gpu(i,1-l,k,2) = -rho*uu
                        w_gpu(i,1-l,k,3) = rho*(2.*jet_velocity  - vv)
                        w_gpu(i,1-l,k,4) = -rho*ww
                        w_gpu(i,1-l,k,5) = pp*gm+qq*rho
                    enddo
#else
                    jet_velocity = blowing_bc_slot_velocity(i-x_start_slot+1,k)

                    ! w() is indexed differently on CPU and GPU
                    rho = w(1,i,j,k)
                    w(3, i,j,k) = rho * jet_velocity
#endif
                enddo
            enddo
            !@cuf iercuda=cudaDeviceSynchronize()
        endif
    else
        ! TODO: error - we should not be here
    endif

end subroutine bcblow

! given parameters from the input file
! slot_start_x_global
! slot_end_x_global
!
! we calculate LOCALLY:
! nx_slot (the length of the slot as it pertains to OUR mpi node, used for array allocations)
! nz_slot = nz, the width of the slot into the channel
! x_start_slot (the LOCAL location on this MPI node where the blowing boundary condition should be applied)
subroutine local_slot_locations()
    use mod_streams, only: slot_start_x_global, slot_end_x_global,  nx_slot, &
        nz_slot, x_start_slot, nrank, nz, masterproc, nx, force_sbli_blowing_bc 

    implicit none

    integer :: mpi_xstart, mpi_xend
    integer :: true_slot_start, slot_length, slot_end_local, nxmax
    integer :: x_end_slot ! For debugging

    write(*, *) "slot start and slot end (global) is ", slot_start_x_global, slot_end_x_global

    ! the STARTING location in a global reference frame of what grid we are 
    ! responsible for on this MPI node
    mpi_xstart = nx * nrank

    ! the ENDING location in a global reference frame of what grid we are 
    ! responsible for on this MPI node
    mpi_xend = mpi_xstart + nx

    ! for example, with NX = 25 and 4 MPI nodes,
    ! rank 0 is responsible for 1:25
    ! rank 1 is responsible for 26:50
    ! etc...
    ! but above, mpi_xstart = 0 at rank 0, and mpi_xend = 25. This makes the math
    ! later on a bit easier but in principle it is the same thing

    ! if we have a blowing boundary condition
    ! AND
    ! if the end of the slot is AFTER the global starting place of our grid
    ! AND
    ! the start of the slot is BEFORE the global ending point of our grid
    if ( &
        (force_sbli_blowing_bc == 1)  .and. &
        (mpi_xstart <= slot_end_x_global) .and. &
        (slot_start_x_global <= mpi_xend)) then
        ! the start of the blowing slot, relative to the start of 
        ! our grid in a global reference frame. This results in an integer (possibly negative)
        ! on how far forward or backward from the start of our grid the begining of the slot is
        true_slot_start = slot_start_x_global - mpi_xstart

        ! the slot start locally is either the first index (1) OR it is some length past the starting
        ! point of our grid. The start of the slot that we store here CANNOT be negative, because 
        ! that implies that the slot starts within another grid, and we only care about our grid.
        x_start_slot = max(1, true_slot_start)

        ! calculate how long the slot is
        slot_length = slot_end_x_global - slot_start_x_global

        ! then, find out in a global reference frame where the end of the slot is.
        ! We know that LOCALLY the end of the slot is at a maximum NX (since otherwise it would 
        ! run into the next grid)
        slot_end_local = min(nx, true_slot_start + slot_length)

        ! the length of the slot on this local process is just
        ! the local end of the slot minus the local start of the slot
        ! 
        ! then, we have to add one to the slot length for 1-based indexing rules:
        ! if x_start_slot = 1, x_end_slot = 3 then we clearly have 3 locations on the x axis where
        ! we are forcing, but 3-1 = 2, hence the +1
        nx_slot = slot_end_local - x_start_slot + 1
        x_end_slot = slot_end_local

        nz_slot = nz
    else
        x_start_slot = -1
        slot_end_local = -1
        ! these are fairly useless, we wont even allocate an array if 
        ! x_start_slot = -1
        nx_slot = 0
        nz_slot = 0
    endif
 
!    write(*, *) "on nrank ", nrank, "the slot starts at ", x_start_slot, " and ends at ", slot_end_local, &
!        " which is ", nx_slot, " in length, mpi xstart ", mpi_xstart, "mpi xend", mpi_xend, &
!        " force sbli bc is ", force_sbli_blowing_bc

    write (*,'("on nrank ",i0," the slot starts at ",i0," and ends at ",i0, &
            " which is ",i0," in length, mpi xstart ",i0," mpi xend ",i0, &
            " force sbli bc is ",i0)')  &
        nrank, x_start_slot, slot_end_local, nx_slot, mpi_xstart, mpi_xend, &
        force_sbli_blowing_bc

end subroutine local_slot_locations

! copy
! CPU -> GPU
subroutine copy_blowing_bc_to_gpu()
    use mod_streams
    if (x_start_slot /= -1) then

#ifdef USE_CUDA
        blowing_bc_slot_velocity_gpu = blowing_bc_slot_velocity
#else
        call move_alloc(blowing_bc_slot_velocity, blowing_bc_slot_velocity_gpu)
#endif

    endif
end subroutine

! copy
! GPU -> CPU
subroutine copy_blowing_bc_to_cpu()
    use mod_streams

    if (x_start_slot /= -1) then

#ifdef USE_CUDA
        blowing_bc_slot_velocity = blowing_bc_slot_velocity_gpu
#else
        call move_alloc(blowing_bc_slot_velocity_gpu, blowing_bc_slot_velocity)
#endif

    endif
end subroutine

subroutine allocate_blowing_bcs()
    use mod_streams
    if (x_start_slot /= -1) then
        allocate(blowing_bc_slot_velocity(nx_slot, nz_slot))

#ifdef USE_CUDA
        allocate(blowing_bc_slot_velocity_gpu(nx_slot, nz_slot))
#endif
    endif
end subroutine
