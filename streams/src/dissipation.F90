! calculate the dissipation rate. See lab streams documentation for
! `Calculated Quantities` for the equation being calculated
!
! after calculation, the value is stored in `dissipation_rate` in mod_streams
subroutine dissipation_calculation()
    use mod_streams

    implicit none

    integer :: i, j, k

    real*8 :: uu, vv, ww, rho
    real*8 :: dudx, dvdy, dwdz
    real*8 :: dx, dy, dz
    real*8 :: mpi_dissipation_sum

    ! dissipation_rate is defined in mod_streams, we reset its value here
    dissipation_rate = 0.0

    mpi_dissipation_sum = 0.0

    ! w_gpu(:, :, :, 1) -> rho
    ! w_gpu(:, :, :, 2) -> rho u
    ! w_gpu(:, :, :, 3) -> rho v
    ! w_gpu(:, :, :, 4) -> rho w
    ! w_gpu(:, :, :, 5) -> rho E

    !$cuf kernel do(3) <<<*,*>>> 
    do k = 1,nz
        do j = 1,ny
            do i = 1,nx
                !
                ! at i = 1 dx = 0 so the boundaries will be really wonky. 
                ! the same happens for dy at j = 1, and for dz at k = 1
                !
                if (i == 1) then
                    dx = (x_gpu(i+1) - x_gpu(i)) / 2
                else
                    dx = (x_gpu(i+1) - x_gpu(i-1)) / 2
                endif

                if (j == 1) then
                    dy = (y_gpu(j+1) - y_gpu(j)) / 2
                else
                    dy = (y_gpu(j+1) - y_gpu(j-1)) / 2
                endif

                if (k == 1) then
                    dz = (z_gpu(k+1) - z_gpu(k)) / 2
                else
                    dz = (z_gpu(k+1) - z_gpu(k-1)) / 2
                endif

                !
                ! du/dx
                !
                dudx = ( &
                    ( &
                        -1 * w_gpu(i+2, j, k, 2) / w_gpu(i+2, j, k, 1) &
                    ) &
                    + &
                    ( &
                        8 * w_gpu(i+1, j, k, 2) / w_gpu(i+1, j, k, 1) &
                    ) &
                    - &
                    ( &
                        8 * w_gpu(i-1, j, k, 2) / w_gpu(i-1, j, k, 1) &
                    ) &
                    + &
                    ( &
                        w_gpu(i-2, j, k, 2) / w_gpu(i-2, j, k, 1) &
                    ) &
                ) &
                / ( 12 * dx)

                !
                ! dv/dy
                !
                ! fdm_y_stencil_gpu coeffs are calculated in generate_full_fdm_stencil() subroutine
                dvdy = &
                    (w_gpu(i, j, k, 3) / w_gpu(i, j, k, 1)) * fdm_y_stencil_gpu(1, j) + &
                    (w_gpu(i, j+1, k, 3) / w_gpu(i, j+1, k, 1)) * fdm_y_stencil_gpu(2, j) + &
                    (w_gpu(i, j-1, k, 3) / w_gpu(i, j-1, k, 1)) * fdm_y_stencil_gpu(3, j) + &
                    (w_gpu(i, j+2, k, 3) / w_gpu(i, j+2, k, 1)) * fdm_y_stencil_gpu(4, j) + &
                    (w_gpu(i, j-2, k, 3) / w_gpu(i, j-2, k, 1)) * fdm_y_stencil_gpu(5, j)

                !
                ! dw/dz
                !
                dwdz = ( &
                    ( &
                        -1 * w_gpu(i, j, k+2, 4) / w_gpu(i, j, k+2, 1) &
                    ) &
                    + &
                    ( &
                        8 * w_gpu(i, j, k+1, 4) / w_gpu(i, j, k+1, 1) &
                    ) &
                    - &
                    ( &
                        8 * w_gpu(i, j, k-1, 4) / w_gpu(i, j, k-1, 1) &
                    ) &
                    + &
                    ( &
                        w_gpu(i, j, k-2, 4) / w_gpu(i, j, k-2, 1) &
                    ) &
                ) &
                / ( 12 * dz)

                dissipation_rate = dissipation_rate + &
                    ((dudx**2 + dvdy**2 + dwdz**2) * dx * dy * dz)
            enddo
        enddo
    enddo
    !@cuf iercuda=cudaDeviceSynchronize()

    dissipation_rate = dissipation_rate / (rlx * rly * rlz)

    !write(*, *) "dissipation rate", dissipation_rate

    ! sum all the values across MPI, store the result in mpi_dissipation_sum
    call MPI_REDUCE(dissipation_rate, mpi_dissipation_sum, 1, mpi_prec,  MPI_SUM, 0, MPI_COMM_WORLD, iermpi)

    ! distribute mpi_dissipation_sum to all MPI procs
    call MPI_BCAST(mpi_dissipation_sum, 1, mpi_prec, 0, MPI_COMM_WORLD, iermpi)

    dissipation_rate = mpi_dissipation_sum
endsubroutine dissipation_calculation

subroutine generate_full_fdm_stencil()
    use mod_streams, only: delta => fdm_individual_stencil, alpha => fdm_grid_points, &
        ny, fdm_y_stencil, fdm_y_stencil_gpu, masterproc, y

    implicit none

    integer :: j, i
    real*8 :: y0, ym1, ym2, yp1, yp2, total
    total = 0.0

    if (masterproc) write(*,*) "generating FDM stencil for dissipation"

    do j = 1,ny
        ym2 = y(j-2)
        ym1 = y(j-1)
        y0  = y(j)
        yp1 = y(j+1)
        yp2 = y(j+2)

        ! j
        alpha(0) = y0
        ! j+1
        alpha(1) = yp1
        ! j-1
        alpha(2) = ym1
        ! j+2
        alpha(3) = yp2
        ! j-2
        alpha(4) = ym2

        ! use alpha to compute the stencil for this set of points
        call irregular_grid_stencil()
        fdm_y_stencil(:, j) = delta(1, 4, :)

        do i = 1,5
            total = total + abs(fdm_y_stencil(i, j))
        enddo
    enddo

    if (masterproc) write(*,*) "total of stencil coefficients is", total

! copy over to GPU variable
#ifdef USE_CUDA
    fdm_y_stencil_gpu = fdm_y_stencil
#else
    call move_alloc(fdm_y_stencil, fdm_y_stencil_gpu)
#endif

endsubroutine

! fetches the grid points from `fdm_grid_points` and operates on `fdm_individual_stencil_gpu` 
! to perform the algorithm from
! Generation of Finite Difference Formulas on Arbitrarily Spaced Grids (Fornberg, 1988)
! to generate a stencil that we can use for the finite difference calculation
subroutine irregular_grid_stencil()
    use mod_streams, only: delta => fdm_individual_stencil, alpha => fdm_grid_points

    implicit none

    integer :: M0, N0, v, m, n
    real*8 :: c1, c2, c3, t1, t2, t3, x0

    M0 = 1
    N0 = 4

    delta(:, :, :) = 0.0
    delta(0, 0, 0) = 1.0
    c1 = 1.0

    x0 = alpha(0)

    do n = 1,N0
        c2 = 1.0

        do v = 0,n-1
            c3 = alpha(n) - alpha(v)
            c2 = c2 * c3

            do m = 0, min(n,M0)
                t3 = delta(m, n-1, v)
                t1 = (alpha(n) - x0) * t3

                if (m == 0) then
                    t2 = 0.0
                else
                    t2 = m * delta(m-1, n-1, v)
                endif

                delta(m, n, v)  = (t1 -t2) / c3
            enddo
        enddo

        do m = 0, min(n,M0)
            if (m == 0) then
                t1 = 0.0
            else 
                t1 = m * delta(m-1, n-1, n-1)
            endif

            t3 = delta(m, n-1, n-1)
            t2 = (alpha(n-1) - x0) * t3

            delta(m, n, n) = (c1 / c2) * (t1 - t2)
        enddo

        c1 = c2
    enddo

end subroutine

! calculate the first order derivative with 4th order error for a simple grid
subroutine validate_irregular_fdm()
    use mod_streams

    fdm_grid_points(0) = 0.0
    fdm_grid_points(1) = 1.
    fdm_grid_points(2) = -1.
    fdm_grid_points(3) = 2.
    fdm_grid_points(4) = -2.

    call irregular_grid_stencil()
    ! should output 
    ! -0.0          (0)
    ! 0.666667      (2/3)
    ! -0.666667     (-2/3)
    ! -0.0833333    (-1/12)
    ! 0.0833333     (1/12)
#ifdef USE_CUDA
#else
    ! can only print in CPU mode
    write(*,*) fdm_individual_stencil(1, 4, :)
#endif
endsubroutine

subroutine energy_calculation()
    use mod_streams

    implicit none

    integer :: i, j, k

    real*8 :: uu, vv, ww, rho
    real*8 :: dudx, dvdy, dwdz
    real*8 :: dx, dy, dz
    real*8 :: mpi_energy_sum

    ! dissipation_rate is defined in mod_streams, we reset its value here
    energy = 0.0

    mpi_energy_sum = 0.0

    ! w_gpu(:, :, :, 1) -> rho
    ! w_gpu(:, :, :, 2) -> rho u
    ! w_gpu(:, :, :, 3) -> rho v
    ! w_gpu(:, :, :, 4) -> rho w
    ! w_gpu(:, :, :, 5) -> rho E

    !$cuf kernel do(3) <<<*,*>>> 
    do k = 1,nz
        do j = 1,ny
            do i = 1,nx
                rho = w_gpu(i, j, k, 1)
                uu = w_gpu(i, j, k, 2) / rho
                vv = w_gpu(i, j, k, 3) / rho
                ww = w_gpu(i, j, k, 4) / rho

                dx = (x_gpu(i+1) - x_gpu(i-1)) / 2
                dy = (y_gpu(j+1) - y_gpu(j-1)) / 2
                dz = (z_gpu(k+1) - z_gpu(k-1)) / 2

                energy = energy + &
                    (uu**2 + vv**2 + ww**2) * dx * dy * dz
            enddo
        enddo
    enddo

    energy = energy * 0.5

    ! sum all the values across MPI, store the result in mpi_energy_sum 
    call MPI_REDUCE(energy, mpi_energy_sum, 1, mpi_prec,  MPI_SUM, 0, MPI_COMM_WORLD, iermpi)

    ! distribute mpi_energy_sum to all MPI procs
    call MPI_BCAST(mpi_energy_sum, 1, mpi_prec, 0, MPI_COMM_WORLD, iermpi)

    energy = mpi_energy_sum

end subroutine energy_calculation
