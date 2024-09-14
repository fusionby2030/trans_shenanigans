MODULE PHYSICS 
    use types_and_kinds
    use helpers
    IMPLICIT NONE
CONTAINS
SUBROUTINE computetimestep(sim, dt)
    ! CFL condition for diffusion equation
    TYPE(Simulation), INTENT(IN) :: sim
    REAL(DP), INTENT(OUT) :: dt
    REAL(DP) :: dx, maxchi, maxD, maxV
    dx = sim%dx ! 1.0_DP / (sim%nx-1)
    maxchi = MAXVAL(sim%transparams%chi)
    maxD = MAXVAL(sim%transparams%D)
    maxV = MAXVAL(sim%transparams%V)
    dt = (dx*dx / (4.0_DP * MAX(maxchi, maxD + maxV)))
    ! print *, dt
END SUBROUTINE computetimestep
SUBROUTINE update(sim, dt)
    ! Forward Euler Update
    type(simulation), INTENT(INOUT) :: sim
    REAL(DP), INTENT(IN) :: dt
    REAL(DP), DIMENSION(sim%nx+2*sim%nghosts) :: T, n, chi, D, V, S_T, S_N, x
    REAL(DP), DIMENSION(sim%nx+2*sim%nghosts) :: d2Tdx2, d2ndx2, dTdx, dndx, dChidx, dDdx, dVdx
    REAL(DP) :: dx
    INTEGER(IP) :: i
    dx = sim%dx
    x = sim%grid%psin
    T = sim%prim%T
    n = sim%prim%n
    chi = sim%transparams%chi 
    D = sim%transparams%D
    V = sim%transparams%V
    S_T = sim%transparams%S_T
    S_N = sim%transparams%S_N

    do i=1+sim%nghosts, sim%nx+sim%nghosts
        dTdx(i) = (T(i+1) - T(i-1)) / (2.0_DP*dx)
        dndx(i) = (n(i+1) - n(i-1)) / (2.0_DP*dx)
        dChidx(i) = (chi(i+1) - chi(i-1)) / (2.0_DP*dx)
        dDdx(i) = (D(i+1) - D(i-1)) / (2.0_DP*dx)
        dVdx(i) = (V(i+1) - V(i-1)) / (2.0_DP*dx)
    end do
    do i=1+sim%nghosts, sim%nx+sim%nghosts
        d2Tdx2(i) = (T(i+1) - 2.0_DP*T(i) + T(i-1)) / (dx*dx)
        d2ndx2(i) = (n(i+1) - 2.0_DP*n(i) + n(i-1)) / (dx*dx)
    end do
    do i=1+sim%nghosts, sim%nx+sim%nghosts
        T(i) = T(i) + dt * (x(i) * (chi(i) * d2Tdx2(i) + dChidx(i)*dTdx(i)) + chi(i)*dTdx(i) + S_T(i))
        n(i) = n(i) + dt * (D(i) * d2ndx2(i) + V(i)*dndx(i) + dDdx(i)*dndx(i) + dndx(i)*dVdx(i) + S_N(i))
    end do
    sim%prim%T = T
    sim%prim%n = n
    ! Pressure is 2*n*t*boltzmann
    sim%derived%P = (2.0_DP *(sim%prim%T*11604.5_DP) * (n / 10.0_DP)  * 1.38064852_DP) / 1000.0_DP
END SUBROUTINE update
SUBROUTINE initialize_simulation(sim)
    TYPE(SIMULATION), INTENT(INOUT) :: sim 
    ALLOCATE(sim%prim%T(sim%nx+2*sim%nghosts), sim%prim%n(sim%nx+2*sim%nghosts))
    ALLOCATE(sim%grid%psin(sim%nx+2*sim%nghosts))
    ALLOCATE(sim%derived%p(sim%nx+2*sim%nghosts), sim%derived%alpha(sim%nx+2*sim%nghosts))
    ALLOCATE(sim%transparams%chi(sim%nx+2*sim%nghosts), sim%transparams%D(sim%nx+2*sim%nghosts), sim%transparams%V(sim%nx+2*sim%nghosts), sim%transparams%S_T(sim%nx+2*sim%nghosts), sim%transparams%S_N(sim%nx+2*sim%nghosts))

    sim%grid%psin = 0.0_DP
    CALL LINSPACE_RETSTEP(sim%grid%psin, 0.8_DP, 1.0_DP, sim%dx)

    ! Pedestal idx is the index where sim&pedestal_loc is closest to psin grid
    sim%pedestal_idx = MINLOC(ABS(sim%grid%psin - sim%pedestal_loc), 1)

    call LINEAR_GAUSSIAN(sim%grid%psin(1+sim%nghosts:sim%nx+sim%nghosts), sim%transparams%S_T(1+sim%nghosts:sim%nx+sim%nghosts), sim%power_input, 0.4_DP, 0.2_DP, 0.8_DP)
    CALL MTANH(sim%grid%psin, sim%prim%T, 0.50173193_DP, 0.0_DP, 0.14852205_DP, 0.98343676_DP, 0.04383409_DP)
    call MTANH(sim%grid%psin, sim%prim%n, 3.44198_DP, 0.0_DP, 0.030139_DP, 0.98343676_DP, 0.05961_DP)
    ! call MTANH(sim%grid%psin, sim%prim%n, 3.44198_DP, 0.0_DP, 0.030139_DP, sim%pedestal_loc, sim%pedestal_width)

    ! CALL RSQUARED(sim%grid%psin, sim%prim%n, 4.0_DP)
    ! "prof_height, prof_slope, prof_position, prof_width = 3.44198, 0.030139,0.99695,0.05961
    sim%prim%n(sim%nx + 2*sim%nghosts) = 0.0_DP
    sim%transparams%D   = 0.0_DP ! sim%chi_0
    sim%transparams%V   = 0.0_DP
    sim%transparams%S_N = 0.0_DP ! 20.0_DP
    sim%transparams%Chi = 0.5_DP    
END SUBROUTINE initialize_simulation
SUBROUTINE update_ghosts_and_bcs(sim)
    TYPE(Simulation), INTENT(INOUT) :: sim
    sim%prim%T(1)                    = sim%prim%T(2)
    sim%prim%T(sim%nx+sim%nghosts)   = 0.1 ! sim%prim%T(sim%nx + sim%nghosts)
    sim%prim%T(sim%nx+1+sim%nghosts) = sim%prim%T(sim%nx + sim%nghosts)

    sim%prim%n(1) = sim%prim%n(2)
    sim%prim%n(sim%nx+1+sim%nghosts) = sim%prim%n(sim%nx + sim%nghosts)
    sim%prim%n(sim%nx+2*sim%nghosts) = 2.0_DP*sim%prim%n(sim%nx+ 2*sim%nghosts -1) - sim%prim%n(sim%nx)


    sim%transparams%chi(1) = sim%transparams%chi(2)
    sim%transparams%chi(sim%nx+1+sim%nghosts) = sim%transparams%chi(sim%nx + sim%nghosts)
END SUBROUTINE update_ghosts_and_bcs
SUBROUTINE update_transparams(sim)
    TYPE(SIMULATION), INTENT(INOUT) :: SIM 
    REAL(DP), DIMENSION(sim%nx+2*sim%nghosts) :: chi, D, V, x, gradT, gradN
    INTEGER(IP) :: i

    chi = sim%transparams%chi
    D   = sim%transparams%D
    V   = sim%transparams%V
    x   = sim%grid%psin

    gradT = 0.0_DP 
    gradN = 0.0_DP 
    do i=1+sim%nghosts, sim%nx+sim%nghosts
        gradT(i) = (sim%prim%T(i+1) - sim%prim%T(i-1)) / (2.0_DP*sim%dx)
        gradN(i) = (sim%prim%n(i+1) - sim%prim%n(i-1)) / (2.0_DP*sim%dx)
    end do

    if (sim%intra_elm_active .eqv. .TRUE.) then 
        call chi_intra_model(sim, gradT, chi)
        ! D
        ! V
    else 
        call chi_inter_model(sim, gradT, chi)
        ! call chi_model_emi(chi, x, sim%prim%T, gradT, sim%pedestal_loc, sim%c_etb, sim%chi_0, sim%tgrad_crit)
        chi = max(chi, 0.001_DP)
        ! D 
        ! V
    end if 
    sim%transparams%chi = chi
    sim%transparams%D   = D
    sim%transparams%V   = V
END SUBROUTINE update_transparams
SUBROUTINE chi_model_emi(chi, x, T, gradT, pedestal_loc, cx, chi_0, Tgrad_crit)
    REAL(DP), INTENT(IN) :: x(:), T(:), gradT(:), chi_0, Tgrad_crit
    REAL(DP), INTENT(INOUT) :: chi(:)
    REAL(DP) :: factor_pedestal_chi(size(x))
    REAL(DP), INTENT(IN) :: pedestal_loc, cx
    ! pedestal_loc = 0.98343676_DP  - 0.04383409_DP
    ! cx = 0.015_DP

    call critical_gradient_core(chi, gradT, chi_0, Tgrad_crit)
    call CHI_PED_FACTOR(x, pedestal_loc, cx, factor_pedestal_chi)
    chi = chi * factor_pedestal_chi
END SUBROUTINE chi_model_emi
SUBROUTINE CHI_PED_FACTOR(r, pedestal_loc, cx, factor_pedestal_chi)
    REAL(DP), INTENT(IN) :: r(:)
    REAL(DP), INTENT(IN) :: pedestal_loc, cx
    REAL(DP), INTENT(OUT) :: factor_pedestal_chi(:)
    REAL(DP) :: factor_at_pedestal, val, mu_val
    INTEGER(IP) :: i
    mu_val = 1.0
    call NORMAL_REAL(pedestal_loc, 1.0_DP, cx, factor_at_pedestal)
    do i=1, size(r)
        if (r(i) >= pedestal_loc) then
            call NORMAL_REAL(r(i), 1.0_DP, cx, val)
            factor_pedestal_chi(i) = factor_at_pedestal / val
        else
            factor_pedestal_chi(i) = 1.0_DP
        end if
    end do
END SUBROUTINE CHI_PED_FACTOR

SUBROUTINE chi_intra_model(sim, gradT, chi)
    TYPE(SIMULATION), INTENT(IN) :: sim
    REAL(DP), DIMENSION(sim%nx+2*sim%nghosts), INTENT(IN) :: gradT
    REAL(DP), DIMENSION(sim%nx+2*sim%nghosts), INTENT(OUT) :: chi
    REAL(DP) :: amplitude
    ! TODO: move pedestal loc...
    amplitude = sim%c_crash / (SQRT(2*PI*sim%pedestal_width**2))
    call GAUSSIAN(sim%grid%psin, chi, amplitude, sim%pedestal_loc + sim%pedestal_width, sim%pedestal_width)
    chi = chi + sim%chi_0
END SUBROUTINE chi_intra_model

SUBROUTINE chi_inter_model(sim, gradT, chi)
    TYPE(SIMULATION), INTENT(IN) :: sim
    REAL(DP), DIMENSION(sim%nx+2*sim%nghosts), INTENT(IN) :: gradT
    REAL(DP), DIMENSION(sim%nx+2*sim%nghosts), INTENT(INOUT) :: chi
    call critical_gradient_core(chi, gradT, sim%chi_0, sim%tgrad_crit)    
    call CHI_ETB_BEZIER(chi(sim%pedestal_idx), sim%pedestal_loc, 0.001_DP, 1.0_DP, sim%c_etb, sim%grid%psin, chi)
    ! Make sure it is always positive 
    ! chi = MAX(chi, 0.0_DP)
END SUBROUTINE chi_inter_model

SUBROUTINE critical_gradient_core(chi, gradT, chi_0, Tgrad_crit)
    REAL(DP), INTENT(IN) :: gradT(:), chi_0, Tgrad_crit! =1.0_DP
    REAL(DP), INTENT(INOUT) :: chi(:)
    REAL(DP) :: a=0.5_DP, k=2.0_DP
    ! REAL(DP) :: Tgrad_crit = 0.2_DP
    INTEGER(IP) :: i
    ! chi = chi_0
    chi = chi_0
    do i=1, size(chi)
        if (abs(gradT(i)) > abs(Tgrad_crit)) then
            chi(i) = k*(abs(gradT(i)) - Tgrad_crit)**a + chi_0
            ! print *, "Changing chi at ", i, gradT(i), chi(i)
        end if
    end do
END SUBROUTINE critical_gradient_core

    
SUBROUTINE CHI_ETB_BEZIER(CHI_PED, X_PED, CHI_SEP, X_SEP, c_etb, grid, new_chi)
    REAL(DP), INTENT(IN) :: CHI_PED, X_PED, CHI_SEP, X_SEP, c_etb, grid(:)
    REAL(DP), INTENT(INOUT) :: new_chi(:)
    INTEGER(IP), PARAMETER :: NUM_T = 1000
    REAL(DP) :: P1(2), P2(2), P3(2), X_VALUES(NUM_T), Y_VALUES(NUM_T)
    INTEGER :: I 
    REAL(DP) :: mid_x, mid_y

    ! Calculate midpoints and control points 
    mid_x = (X_PED + X_SEP) / 2.0_DP
    mid_y = (CHI_PED + CHI_SEP) / 2.0_DP
    ! mid_x = 0.98343676_DP - 0.04383409_DP/2.0_DP
    P1    = [X_PED, CHI_PED]
    P3    = [X_SEP, CHI_SEP]

    P2(1) = mid_x + c_etb*(X_PED - mid_x)
    P2(2) = mid_y + c_etb*(CHI_SEP - mid_y)

    CALL QUAD_BEZIER(P1, P2, P3, NUM_T, X_VALUES, Y_VALUES)

    ! Interpolate onto the psin grid 
    DO I=1, size(grid)
        if (grid(i) < X_PED) then 
            ! pass and continue 
            ! print *, "keeping val at ", i, grid(i), new_chi(i)
        else 
            if (grid(i) <= X_VALUES(1)) then 
                new_chi(i) = y_values(i) 
            else if (grid(i) >= x_values(NUM_T)) then 
                new_chi(i) = y_values(NUM_T)
            else 
                ! print *, "Changing val at x", grid(i) 
                ! Linear interpolation between closest points 
                call linear_interpolate(grid(i), x_values, y_values, NUM_T, new_chi(i))
            end if
        end if 
    END DO 
END SUBROUTINE CHI_ETB_BEZIER
SUBROUTINE mhd_stability_approximation(sim, t_lastelm)
    TYPE(Simulation), INTENT(INOUT) :: sim
    REAL(DP) :: psin_maxalpha, val_maxalpha
    REAL(DP), INTENT(INOUT) :: t_lastelm
    INTEGER(IP) :: idx_max_alpha
    INTEGER(IP) :: i
    ! Calculate pressure gradient and compute alpha
    sim%derived%alpha = 0.0_DP
    do i=1+sim%nghosts, sim%nx+sim%nghosts
        sim%derived%alpha(i) = (sim%derived%p(i+1) - sim%derived%p(i-1)) / (2*sim%dx)
    end do
    ! sim%derived%alpha = abs(sim%derived%alpha) / abs(MAXVAL(sim%derived%alpha))
    sim%derived%alpha = abs(sim%derived%alpha)
    CALL find_maxval_alpha_in_pedestal(sim, idx_max_alpha, val_maxalpha, psin_maxalpha)

    if ((sim%intra_elm_active .eqv. .True.) .AND. (t_lastelm < 0.0002_DP)) then
        sim%intra_elm_active = .True.
    else if ((sim%intra_elm_active .eqv. .True.) .AND. (t_lastelm >= 0.0002_DP)) then
        sim%intra_elm_active = .False.
    else
        IF (val_maxalpha > sim%pressure_grad_threshold) THEN
            sim%intra_elm_active = .TRUE.
            t_lastelm = 0.0_DP
        ELSE
            sim%intra_elm_active = .FALSE.
        END IF
    end if
    ! print *, val_maxalpha, psin_maxalpha, idx_max_alpha
END SUBROUTINE mhd_stability_approximation
SUBROUTINE find_maxval_alpha_in_pedestal(sim, idx_max_alpha, val_maxalpha, psin_maxalpha)
    TYPE(Simulation), INTENT(IN) :: sim
    INTEGER(IP), INTENT(OUT) :: idx_max_alpha
    REAL(DP), INTENT(OUT) :: val_maxalpha, psin_maxalpha
    INTEGER(IP) :: i
    val_maxalpha = 0.0_DP
    do i=1+sim%nghosts, sim%nx+sim%nghosts
        ! Check if in pedestal by comparing psin to sim%pedestal_loc
        if (sim%grid%psin(i) > sim%pedestal_loc) then
            if (sim%derived%alpha(i) > val_maxalpha) then
                idx_max_alpha = i
                val_maxalpha = sim%derived%alpha(i)
                psin_maxalpha = sim%grid%psin(i)
            end if
        end if
    end do
END SUBROUTINE find_maxval_alpha_in_pedestal
END MODULE PHYSICS 
