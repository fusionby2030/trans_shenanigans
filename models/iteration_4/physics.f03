MODULE PHYSICS
    use types_and_kinds
    use helpers
    IMPLICIT NONE
CONTAINS
SUBROUTINE update_ghosts_and_bcs(sim)
    TYPE(Simulation), INTENT(INOUT) :: sim
    sim%prim%T(1)                    = sim%prim%T(2)
    sim%prim%T(sim%nx+sim%nghosts)   = 0.1 ! sim%prim%T(sim%nx + sim%nghosts)
    sim%prim%T(sim%nx+1+sim%nghosts) = sim%prim%T(sim%nx + sim%nghosts)

    sim%prim%n(1) = sim%prim%n(2)
    sim%prim%n(sim%nx+1+sim%nghosts) = sim%prim%n(sim%nx + sim%nghosts)
    ! Want second derivative of density to be 0 at RHS boundary
    ! (n(i+1) - 2.0_DP*n(i) + n(i-1)) / (dx*dx) = 0 -> n(GHOST) = 2.0_DP*n(nx) - n(nx-1)
    sim%prim%n(sim%nx+2*sim%nghosts) = 2.0_DP*sim%prim%n(sim%nx+ 2*sim%nghosts -1) - sim%prim%n(sim%nx)


    sim%transparams%chi(1) = sim%transparams%chi(2)
    sim%transparams%chi(sim%nx+1+sim%nghosts) = sim%transparams%chi(sim%nx + sim%nghosts)

END SUBROUTINE update_ghosts_and_bcs
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
    REAL(DP) FUNCTION find_maxval_alpha_in_pedestal_func(sim)
        TYPE(Simulation), INTENT(IN) :: sim
        INTEGER(IP) :: i
        REAL(DP) :: maxval_alpha
        maxval_alpha = 0.0_DP
        do i=1+sim%nghosts, sim%nx+sim%nghosts
            ! Check if in pedestal by comparing psin to sim%pedestal_loc
            if (sim%grid%psin(i) > sim%pedestal_loc) then
                maxval_alpha = MAX(maxval_alpha, sim%derived%alpha(i))
            end if
        end do
        find_maxval_alpha_in_pedestal_func = maxval_alpha
    END FUNCTION find_maxval_alpha_in_pedestal_func

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
        chi = sim%transparams%chi ! This might be really stupid...
        ! CALL chi_model_emi(chi, x, T, gradT, sim%pedestal_loc, sim%c_etb, sim%chi_0)
        D = sim%transparams%D
        V = sim%transparams%V
        S_T = sim%transparams%S_T
        S_N = sim%transparams%S_N
        ! Update interior cells
        ! Compute first order central differnces for gradients
        do i=1+sim%nghosts, sim%nx+sim%nghosts
            dTdx(i) = (T(i+1) - T(i-1)) / (2.0_DP*dx)
            dndx(i) = (n(i+1) - n(i-1)) / (2.0_DP*dx)
            dChidx(i) = (chi(i+1) - chi(i-1)) / (2.0_DP*dx)
            dDdx(i) = (D(i+1) - D(i-1)) / (2.0_DP*dx)
            dVdx(i) = (V(i+1) - V(i-1)) / (2.0_DP*dx)
        end do
        ! Compute second order central differnces for temp and density
        do i=1+sim%nghosts, sim%nx+sim%nghosts
            d2Tdx2(i) = (T(i+1) - 2.0_DP*T(i) + T(i-1)) / (dx*dx)
            d2ndx2(i) = (n(i+1) - 2.0_DP*n(i) + n(i-1)) / (dx*dx)
        end do
        ! Update interior cells
        do i=1+sim%nghosts, sim%nx+sim%nghosts
            T(i) = T(i) + dt * (x(i) * (chi(i) * d2Tdx2(i) + dChidx(i)*dTdx(i)) + chi(i)*dTdx(i) + S_T(i))
            n(i) = n(i) + dt * (D(i) * d2ndx2(i) + V(i)*dndx(i) + dDdx(i)*dndx(i) + dndx(i)*dVdx(i) + S_N(i))
        end do
        sim%prim%T = T
        sim%prim%n = n
        ! Pressure is 2*n*t*boltzmann
        sim%derived%P = (2.0_DP *(sim%prim%T*11604.5_DP) * (n / 10.0_DP)  * 1.38064852_DP) / 1000.0_DP
    END SUBROUTINE update
    SUBROUTINE chi_intra_elm(sim, gradT, chi)
        TYPE(Simulation), INTENT(IN) :: sim
        REAL(DP), INTENT(IN) :: gradT(:)
        REAL(DP), INTENT(INOUT) :: chi(:)
        INTEGER(IP) :: i
        REAL(DP) :: factor_pedestal_chi(size(gradT))
        ! TODO: move pedestal loc...
        call critical_gradient_core(chi, gradT, sim%chi_0)
        ! call CHI_PED_FACTOR(sim%grid%psin, sim%pedestal_loc, sim%c_crash, factor_pedestal_chi)
        ! factor_pedestal_chi = 1.0_DP / factor_pedestal_chi
        ! chi = chi*factor_pedestal_chi
    END SUBROUTINE chi_intra_elm
    SUBROUTINE D_interelm(sim, D)
        TYPE(Simulation), INTENT(IN) :: sim
        REAL(DP), INTENT(INOUT) :: D(:)
        INTEGER(IP) :: i
        REAL(DP) :: factor_pedestal_D(size(D))
        call CHI_PED_FACTOR(sim%grid%psin, sim%pedestal_loc, sim%c_etb, factor_pedestal_D)
        ! factor_pedestal_D = 1.0_DP / factor_pedestal_D
        D = D*factor_pedestal_D
    END SUBROUTINE D_interelm
    SUBROUTINE chi_model_emi(chi, x, T, gradT, pedestal_loc, cx, chi_0)
        REAL(DP), INTENT(IN) :: x(:), T(:), gradT(:), chi_0
        REAL(DP), INTENT(INOUT) :: chi(:)
        REAL(DP) :: factor_pedestal_chi(size(x))
        REAL(DP), INTENT(IN) :: pedestal_loc, cx
        ! pedestal_loc = 0.98343676_DP  - 0.04383409_DP
        ! cx = 0.015_DP
        call critical_gradient_core(chi, gradT, chi_0)
        call CHI_PED_FACTOR(x, pedestal_loc, cx, factor_pedestal_chi)
        chi = chi * factor_pedestal_chi
    END SUBROUTINE chi_model_emi
    SUBROUTINE critical_gradient_core(chi, gradT, chi_0)
        REAL(DP), INTENT(IN) :: gradT(:), chi_0! =1.0_DP
        REAL(DP), INTENT(INOUT) :: chi(:)
        REAL(DP) :: a=0.5_DP, k=2.0_DP
        REAL(DP) :: Tgrad_crit = 0.2_DP
        INTEGER(IP) :: i
        do i=1, size(chi)
            if (abs(gradT(i)) > abs(Tgrad_crit)) then
                chi(i) = k*(abs(gradT(i)) - Tgrad_crit)**a + chi_0
            end if
        end do
    END SUBROUTINE critical_gradient_core
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
    ! Transport parameters, updating and setting

    SUBROUTINE update_transparams(sim)
        TYPE(Simulation), INTENT(INOUT) :: sim
        REAL(DP), DIMENSION(sim%nx+2*sim%nghosts) :: chi, D, V, x, gradT, gradN
        REAL(DP) :: dx
        INTEGER(IP) :: i
        dx = sim%dx
        x = sim%grid%psin
        chi = sim%chi_0
        D = sim%chi_0 + 2.0_DP

        do i=2, sim%nx+sim%nghosts
            gradT(i) = (sim%prim%T(i+1) - sim%prim%T(i-1)) / (2.0_DP*dx)
            gradN(i) = (sim%prim%n(i+1) - sim%prim%n(i-1)) / (2.0_DP*dx)
        end do
        ! Update parameters based on if ELM is triggered or not
        if (sim%intra_elm_active .eqv. .True.) then
            ! call chi_intra_elm(sim, chi, gradT)
            call critical_gradient_core(chi, gradT, sim%chi_0)
            V = 1.0_DP
        else
            ! Update chi
            call chi_model_emi(chi, x, sim%prim%T, gradT, sim%pedestal_loc, sim%c_etb, sim%chi_0)
            ! Update D
            call D_interelm(sim, D)
            V = x**5
        end if
        sim%transparams%chi = chi
        sim%transparams%D = D
        sim%transparams%V = V
    END SUBROUTINE update_transparams
END MODULE PHYSICS
