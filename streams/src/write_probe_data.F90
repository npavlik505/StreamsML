module mod_probe
  implicit none
  real(8), dimension(:,:,:), allocatable :: span_average
  character(len=5) :: nxstr, nystr
end module mod_probe

module write_probe_data_module
  implicit none

  contains

subroutine to_file(filenumber, i,j)
    use mod_streams
    use mod_sys
    implicit none

    integer :: filenumber
    integer :: i, j, k
    real(mykind) :: rho, rhou, rhov, rhow

    do k = 1, nz
        rho =  w(1,i,j,k)
        rhou = w(2,i,j,k)
        rhov = w(3,i,j,k)
        rhow = w(4,i,j,k)

        write(filenumber) rho
        write(filenumber) rhou / rho
        write(filenumber) rhov / rho
        write(filenumber) rhow / rho
    enddo

end subroutine to_file

!! write_spanwise_probe_helper producing binary with every call
!subroutine write_spanwise_probe_helper(probe_number, i, icyc, j1, j2, j3)
!    implicit none
!    integer :: i, probe_number, icyc, filenumber
!    integer :: j1, j2, j3
!    character(len=120) :: filename
!
!    filenumber = 21
!
!    call system("mkdir -p csv_data")
!
!    write(filename, "(A20, I1.1, A1, I5.5, A7)") "csv_data/span_probe_", probe_number, "_", icyc, ".binary"
!    open(filenumber, file=filename, form="unformatted", access="stream")
!
!    ! viscous data
!    call to_file(filenumber, i, j1)
!
!    ! log law data
!    call to_file(filenumber, i, j2)
!
!    ! free stream data
!    call to_file(filenumber, i, j3)
!
!    call flush(filenumber)
!    close(filenumber)
!
!end subroutine write_spanwise_probe_helper

! write_spanwise_probe_helper producing single binary with data from all calls
subroutine write_spanwise_probe_helper(probe_number, i, icyc, j1, j2, j3)
    implicit none
    integer :: i, probe_number, icyc, filenumber
    integer :: j1, j2, j3
    character(len=120) :: filename

    filenumber = 21

    ! Ensure the directory exists
    call system("mkdir -p csv_data")

    ! Use fixed filenames (remove probe_number and icyc from the filename)
    write(filename, "(A19, I1.1, A15)") "csv_data/span_probe", probe_number, "_viscous.binary"
    open(filenumber, file=filename, form="unformatted", access="stream", status="unknown", position="append")
    ! viscous data
    call to_file(filenumber, i, j1)

    call flush(filenumber)
    close(filenumber)

    ! Repeat for log law data and free stream data using separate fixed filenames

    write(filename, "(A19, I1.1, A26)") "csv_data/span_probe", probe_number, "_just_above_viscous.binary"
    open(filenumber, file=filename, form="unformatted", access="stream", status="unknown", position="append")

    ! log law data
    call to_file(filenumber, i, j2)

    call flush(filenumber)
    close(filenumber)

    write(filename, "(A19, I1.1, A19)") "csv_data/span_probe", probe_number, "_half_height.binary"
    open(filenumber, file=filename, form="unformatted", access="stream", status="unknown", position="append")

    ! free stream data
    call to_file(filenumber, i, j3)

    call flush(filenumber)
    close(filenumber)

end subroutine write_spanwise_probe_helper


subroutine write_probe_data()
    use mod_streams
    use mod_sys

    implicit none
    integer :: i, j, k, dead
    ! the locations in the streamwise direction that 
    ! this mpi process is covering
    integer :: nx_start, nx_end
    ! locations in the y direction that probes will be placed
    integer :: probe_j_1, probe_j_2, probe_j_3
    integer :: probe_i_1, probe_i_2, probe_i_3
    character(len=120) :: filename

    ! probe locations in the i direction
    probe_i_1 = max(int((1./8.)* real(nxmax)), 1)
    probe_i_2 = max(int((4./8.)* real(nxmax)), 1)
    probe_i_3 = max(int((7./8.)* real(nxmax)), 1)
    
    ! probe in the viscous layer (1/70th of the total height)
    probe_j_1 = max(int(real(nymax)      / 70.  ), 1)
    ! probe just above the viscous layer (hopefully log law) (5% of the total height)
    probe_j_2 = max(int(real(nymax) * 5. / 100. ), 1)
    ! probe placed above half of the height (60 / 100)
    probe_j_3 = max(int(real(nymax) * 60. / 100.), 1)

    ! The three probe locations 
    !  |____________________|____________________|___________________|____________________|
    !  |                    |                    |                   |                    |    
    !  |                    |                    |                   |                    |    
    !  |         X          |                   X|                   |          X         |    
    !  |                    |                    |                   |                    |        
    !  |                    |                    |                   |                    |    
    !  |                    |                    |                   |                    |    
    !  |                    |                    |                   |                    |    
    !  |         X          |                   X|                   |          X         |    
    !  |_________X__________|___________________X|___________________|__________X_________|
    ! 
    !  X is a span-wise probe (into the page)

    nx_start = 1 + nrank * nx
    nx_end = nx_start + nx

    ! first probe
    if (nx_start < probe_i_1 .and. probe_i_1< nx_end) then
        call write_spanwise_probe_helper(1,probe_i_1, icyc, probe_j_1,probe_j_2, probe_j_3)
    endif

    ! second probe
    if (nx_start < probe_i_2 .and. probe_i_2< nx_end) then
        call write_spanwise_probe_helper(2,probe_i_2, icyc, probe_j_1,probe_j_2, probe_j_3)
    endif

    ! third probe
    if (nx_start < probe_i_3 .and. probe_i_3 < nx_end) then
        call write_spanwise_probe_helper(3,probe_i_3, icyc, probe_j_1,probe_j_2, probe_j_3)
    endif

end subroutine write_probe_data


subroutine init_write_telaps()
    use mod_streams
    implicit none

    if (masterproc) then
        open(995, file="timesteps.csv", action="write", status="replace")
        write(995, *) "telaps"
    endif

end subroutine init_write_telaps


subroutine write_telaps(telaps_in)
    use mod_streams
    use mod_sys
    implicit none

    real(mykind) :: telaps_in

    if (masterproc) then
        open(995, file="timesteps.csv", action="write", position="append")
        write(995, "(E15.10, A1)") telaps_in, ","
    endif

end subroutine write_telaps

! average out all of the values in a span for the vector of conservative variables w

subroutine write_span_averaged
    use mod_streams
    use mod_sys
    use mod_probe

    ! local variables
    character(len=5) :: current_cycle
    character(len=2) :: mpi_process_number
    character(len=70) :: current_filename

    call system("mkdir -p spans")

    ! works for icyc < 99,999 iterations
    write(current_cycle, "(I5.5)") icyc
    current_filename = "spans/span_average_"//current_cycle//"_average.binary"

    allocate(span_average(nx, ny, 5))
    call helper_average_span(1)
    call helper_average_span(2)
    call helper_average_span(3)
    call helper_average_span(4)
    call helper_average_span(5)

    call write_array_bytes(current_filename)

    ! everything else is deallocated by moving the allocation to span_average 
    ! when writing to vtk
    ! we deallocate here so that we dont error when allocating at the start of 

    ! this subroutine on the second time that it is called
    deallocate(span_average)


end subroutine write_span_averaged

! A helper to average out the values on each of the spans for different variables

! Called from the write_span_averaged subroutine with the name of the file to write to
! and the slice of the variable that we are averaging

subroutine helper_average_span(slice_var)
    use mod_streams
    use mod_sys
    use mod_probe
    implicit none

    ! arguments
    integer:: slice_var

    ! local variables
    real(8) :: current_average
    integer:: i, j, k

    do i = 1,nx
        do j = 1, ny
            ! sum up all of the values on the z axis
            do k = 1, nz
                ! hopefully the compiler can inline each of these branches in the hot loop
                ! otherwise this is painfully slow
                if (slice_var == 1) then
                    ! if we are dealing with rho
                    current_average = current_average + w(slice_var,i,j,k)
                else if (slice_var ==2 .or. slice_var == 3 .or. slice_var == 4) then
                    ! if we are not dealing with rho then we need to divide by it
                    current_average = current_average + (w(slice_var,i,j,k) / w(1,i,j,k))
                else 
                    current_average = current_average + &
                        (w(2,i,j,k) / w(1,i,j,k))**2 + &
                        (w(3,i,j,k) / w(1,i,j,k))**2 + &
                        (w(4,i,j,k) / w(1,i,j,k))**2 
                end if

            enddo
            ! calculate the mean of the data
            ! TODO: hopefully this is not an overflow somewhere
            current_average = current_average / nz

            span_average(i,j, slice_var) = current_average
        enddo
    enddo


end subroutine helper_average_span


subroutine write_array_bytes(filename)
    use mod_streams
    use mod_probe
    implicit none

    ! the number of points in each of the directions
    character(len=16) :: curr_cycle
    character(len=70) :: filename
    ! the number of variables that will be sent over mpi
    integer :: count
    integer :: tag
    integer :: master_id
    integer :: i 

    count = nx * ny * 5
 
    master_id = 0

    if (masterproc) then 
        open(23, file=filename, form="unformatted", access="stream")

        ! write all of the current process' span_average information
        call write_current_array_file(23)

        write(*,*) "(master) wrote data for self"

        ! then, recieve all the information from the other MPI procs and write it to the file

        do i = 1,nproc-1
            write(*,*) "(master) wrote data for proc", i
            tag = i
            call MPI_RECV(span_average(1:nx, 1:ny, 1:5), count, mpi_prec, tag, tag, mpi_comm_world, istatus, iermpi)
            call write_current_array_file(23)
        end do
        
        close(23)
    else
        tag = nrank
        write(*,*) "from ", nrank, "sending data now to master"
        call MPI_SEND(span_average(1:nx, 1:ny, 1:5), count, mpi_prec, master_id, tag, mpi_comm_world, iermpi)
    endif


    close(23)

end subroutine write_array_bytes


! write the entire contents of span_average to a file

subroutine write_current_array_file(filehandle)
    use mod_streams
    use mod_probe
    implicit none

    integer :: filehandle
    integer :: arr_idx, i, j

    do i =1,nx
        do j = 1,ny
            do arr_idx = 1,5
                write(filehandle) span_average(i,j,arr_idx)
            enddo
        enddo
    enddo


end subroutine write_current_array_file
end module write_probe_data_module
