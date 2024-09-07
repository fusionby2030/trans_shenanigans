! Program to model the ELM pedestal evolution
! ---------------- Outline of program ----------------
! PDEs: 1D transport for electron temperature and density
!   - Density:       d/dt (n) = d/dx [ D d/dx (n) + V n] + S_N
!   - Temperature:   d/dt (T) = d/dx [ x Chi d/dx (T)] + S_T
! INTER-ELM transport coefficients
!   - Chi: critical gradient model Jardin (2008) JCP 227, 8769, with pedestal factor for edge transport barrier
!       - Chi = Chi_0 if grad(T) <= critical_grad, else k(grad(T) - critical_grad)**alpha + chi_0
!       - Chi then multiplied by pedestal factor: gaussian distribution with center at just inside separatrix
!   - S_T: Core source, empty at edge
!   - D  : TBD
!   - V  : TBD
!   - S_N: TBD
! INTRA-ELM transport coefficients
! iteration 0  : simply multiply the temperature and density by a factor
! iteration TBD: use iteration 0, but solve inverse problem to determine the coefficients that led profiles there
! -----------------------------------------------------
! ------------------ Program structure ----------------

MODULE types_and_kinds
    IMPLICIT NONE
    INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(15, 307)
    INTEGER, PARAMETER :: ip = SELECTED_INT_KIND(8)
    TYPE, PUBLIC :: primitives
        REAL(DP), DIMENSION(:), ALLOCATABLE :: T, n
    END TYPE primitives
    TYPE, PUBLIC :: derived
        REAL(DP), DIMENSION(:), ALLOCATABLE :: p, alpha
    END TYPE derived
    TYPE, PUBLIC :: domain
        REAL(DP), DIMENSION(:), ALLOCATABLE :: psin
    END TYPE domain
    TYPE, PUBLIC :: coeffs
        REAL(DP), DIMENSION(:), ALLOCATABLE :: chi, D, V, S_T, S_N
    END TYPE coeffs
    TYPE, PUBLIC :: simulation
        INTEGER(IP) :: nx, nghosts
        REAL(DP) :: chi_0, critical_grad, dx
        REAL(DP) :: power_input
        REAL(DP) :: c_etb, c_crash
        REAL(DP) :: pedestal_loc, alpha_critical, pressure_grad_threshold
        LOGICAL :: intra_elm_active
        type(primitives) :: prim
        type(coeffs)     :: transparams
        type(domain)     :: grid
        type(derived)    :: derived
    END TYPE simulation

END MODULE types_and_kinds
MODULE initialization
    USE types_and_kinds

    IMPLICIT NONE
CONTAINS
    SUBROUTINE MTANH(r, u, h1, h0, s, p, w)
        REAL(DP), INTENT(IN) :: r(:)
        REAL(DP), INTENT(OUT) :: u(:)
        REAL(DP), INTENT(IN) :: h1, h0, s, p, w
        REAL(DP) :: x(size(r))
        x = (p - r) / (w/2.0_DP)
        u = h0 + ((h1-h0)/2.0_DP)* ( ((1+s*x)*exp(x) - exp(-x)) / (exp(x) + exp(-x)) + 1)
    END SUBROUTINE MTANH
    SUBROUTINE LINSPACE_RETSTEP(x, a, b, step)
        REAL(DP), INTENT(OUT) :: x(:), step
        REAL(DP), INTENT(IN) :: a, b
        CALL LINSPACE(x, a, b)
        step = (b-a)/(size(x)-1)
    END SUBROUTINE LINSPACE_RETSTEP
    SUBROUTINE LINEAR_GAUSSIAN(r, u, A, mu, sigma, scalefactor)
        REAL(DP), INTENT(IN) :: r(:)
        REAL(DP), INTENT(OUT) :: u(:)
        REAL(DP), INTENT(IN) :: A, mu, sigma, scalefactor
        INTEGER(IP) :: idx_x_max
        call GAUSSIAN(r, u, A, mu, sigma)
        idx_x_max = MAXLOC(u, 1)
        call LINSPACE(u(1:idx_x_max-1), u(idx_x_max-1)*scalefactor, u(idx_x_max-1))
    END SUBROUTINE LINEAR_GAUSSIAN
    SUBROUTINE GAUSSIAN(r, u, A, mu, sigma)
        REAL(DP), INTENT(IN) :: r(:)
        REAL(DP), INTENT(OUT) :: u(:)
        REAL(DP), INTENT(IN) :: A, mu, sigma
        u = A * exp(-(r-mu)**2 / (2.0_DP*sigma**2))
    END SUBROUTINE GAUSSIAN
    SUBROUTINE LINSPACE(x, a, b)
        REAL(DP), INTENT(OUT) :: x(:)
        REAL(DP), INTENT(IN) :: a, b
        INTEGER(IP) :: i
        x = [(a + (b-a)*i/(size(x)-1), i=0, size(x)-1)]
    END SUBROUTINE LINSPACE
    SUBROUTINE RSQUARED(r, u, scalar)
        REAL(DP), INTENT(IN) :: r(:)
        REAL(DP), INTENT(OUT) :: u(:)
        REAL(DP), INTENT(IN) :: scalar
        u = 1.0_DP - r*r
        u = u*scalar
    END SUBROUTINE RSQUARED
    SUBROUTINE NORMAL_REAL(x, mu, sigma, val)
        REAL(DP), INTENT(IN) :: x
        REAL(DP), INTENT(IN) :: mu, sigma
        REAL(DP), INTENT(OUT) :: val
        val = 1.0_DP / (sigma * sqrt(2.0_DP*3.14159265359_DP)) * exp(-(x-mu)**2 / (2.0_DP*sigma**2))
    END SUBROUTINE NORMAL_REAL
END MODULE initialization
MODULE PHYSICS
    use types_and_kinds
    use initialization
    IMPLICIT NONE
CONTAINS
SUBROUTINE update_ghosts_and_bcs(sim)
    TYPE(Simulation), INTENT(INOUT) :: sim
    sim%prim%T(1)                    = sim%prim%T(2)
    sim%prim%T(sim%nx+sim%nghosts)   = 0.1 ! sim%prim%T(sim%nx + sim%nghosts)
    sim%prim%T(sim%nx+1+sim%nghosts) = sim%prim%T(sim%nx + sim%nghosts)

    sim%prim%n(1) = sim%prim%n(2)
    sim%prim%n(sim%nx+1+sim%nghosts) = sim%prim%n(sim%nx + sim%nghosts)

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
        dx = 1.0_DP / (sim%nx-1)
        maxchi = MAXVAL(sim%transparams%chi)
        maxD = MAXVAL(sim%transparams%D)
        maxV = MAXVAL(sim%transparams%V)
        dt = dx*dx / (4.0_DP * MAX(maxchi, maxD + maxV))
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
    SUBROUTINE update_transparams(sim)
        TYPE(Simulation), INTENT(INOUT) :: sim
        REAL(DP), DIMENSION(sim%nx+2*sim%nghosts) :: T, n, chi, D, V, S_T, S_N, x, gradT, gradN, fluxT, fluxN
        REAL(DP) :: dx
        INTEGER(IP) :: i
        dx = sim%dx
        x = sim%grid%psin
        T = sim%prim%T
        n = sim%prim%n
        do i=2, sim%nx+sim%nghosts
            gradT(i) = (T(i+1) - T(i-1)) / (2.0_DP*dx)
            gradN(i) = (n(i+1) - n(i-1)) / (2.0_DP*dx)
        end do
        ! Update parameters based on if ELM is triggered or not
        if (sim%intra_elm_active .eqv. .True.) then
            call critical_gradient_core(chi, gradT, sim%chi_0)
        else
            call chi_model_emi(chi, x, T, gradT, sim%pedestal_loc, sim%c_etb, sim%chi_0)
        end if
        sim%transparams%chi = chi
    END SUBROUTINE update_transparams
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
END MODULE PHYSICS
MODULE plotting_helpers
    use types_and_kinds
    use iso_c_binding, only: c_int, c_int32_t, C_NULL_CHAR, C_NULL_PTR, c_loc, c_float
    use raylib
    use raymath
    IMPLICIT NONE
    INTEGER(c_int), PARAMETER :: ww = 1920, wh = 1280, border_padding=20
    INTEGER(c_int), PARAMETER :: plot_width=0.9*(ww/3), plot_height=0.9*(wh/3)
    REAL(c_float), PARAMETER ::  plot_thicc=3.0
CONTAINS
    SUBROUTINE initialize_primtive_plots(tp, np, coefp, pp)
        TYPE(Rectangle), INTENT(INOUT)  :: tp, np, pp, coefp
        ! Top Left
        tp%x      = border_padding
        tp%y      = border_padding
        tp%width  = plot_width
        tp%height = plot_height
        ! Top Middle
        np%x      = ww / 2 - plot_width / 2
        np%y      = border_padding
        np%width  = plot_width
        np%height = plot_height
        ! Top Right
        coefp%x      = ww - plot_width - border_padding
        coefp%y      = border_padding
        coefp%width  = plot_width
        coefp%height = plot_height
        ! Middle Left
        pp%x      = border_padding
        pp%y      = wh / 2 - plot_height / 2
        pp%width  = plot_width
        pp%height = plot_height
    END SUBROUTINE initialize_primtive_plots
    function round(val, n)
        implicit none
        real(DP) :: val, round
        integer :: n
        round = anint(val*10.0**n)/10.0**n
    end function round
    SUBROUTINE plot_profile_given_subplot(rect, x, y, buff_x, buff_y, y_l_lim, y_u_lim, color)
        TYPE(RECTANGLE), INTENT(IN) :: rect
        REAL(DP), INTENT(IN) :: x(:), y(:)
        REAL, INTENT(IN) :: y_l_lim, y_u_lim
        CHARACTER(len=6) :: charbuff
        INTEGER(c_int32_t), INTENT(IN) :: color
        REAL(DP), INTENT(INOUT) :: buff_x(:), buff_y(:)
        integer(IP) :: i
        REAL(DP) :: max_x, max_y, min_x, min_y

        max_x = 1.0 ! maxval(x) ! x_u_lim ! maxval(x)
        min_x = 0.0 ! minval(x) ! minval(x)
        max_y = y_u_lim ! 2.5 ! maxval(y)
        min_y = y_l_lim ! 0.0 !minval(y)
        buff_x = (max_x - x) / (max_x - min_x )
        buff_y = (max_y - y) / (max_y - min_y)
        buff_x = 1-buff_x
        buff_x = buff_x * rect%width
        buff_y = buff_y * rect%height

        do i=2, size(x) -1
            CALL draw_circle(int(buff_x(i) + rect%x), int(buff_y(i)+ rect%y), 2.0, COLOR)
        end do
        call draw_circle(int(buff_x(1) + rect%x), int(buff_y(1)+ rect%y), 2.0, BLUE)
        call draw_circle(int(buff_x(size(x)) + rect%x), int(buff_y(size(x))+ rect%y), 2.0, BLUE)

        write (charbuff, '(F6.2)') min_x
        call draw_text(charbuff//C_NULL_CHAR, int(rect%x), int(rect%y + rect%height), 20, BLACK)
        write (charbuff, '(F6.2)') max_x
        call draw_text(charbuff//C_NULL_CHAR, int(rect%x + rect%width), int(rect%y + rect%height), 20, BLACK)
        write (charbuff, '(F6.2)') max_y
        call draw_text(charbuff//C_NULL_CHAR, int(rect%x) - border_padding, int(rect%y), 20, BLACK)

    END SUBROUTINE plot_profile_given_subplot
END MODULE plotting_helpers

PROGRAM toy
    use types_and_kinds
    use physics
    use initialization
    use iso_c_binding, only: c_int, c_int32_t, C_NULL_CHAR, C_NULL_PTR, c_loc, c_float
    use raylib
    use raymath
    use plotting_helpers
    IMPLICIT NONE
    TYPE(Simulation) :: sim
    TYPE(Rectangle)  :: temperature_plot, density_plot, pressure_grad_plot, transparam_plot
    character(len=12) :: twritten
    REAL(DP) :: tout=0.0, dt=0.01, wout=0.0, wstep=0.00005, totalsimtime=1.0_dp
    REAL(DP) :: t_lastelm=0.0_DP ! 200 microseconds -> 0.0002
    type(Vector2) :: vec_buffer
    REAL(DP), ALLOCATABLE :: canvas_data_x(:), canvas_data_y(:)
    INTEGER :: int_buffer
    ! ---- Setup ---
    sim%nx = 400
    sim%nghosts = 1
    sim%power_input = 10.5_DP
    sim%intra_elm_active = .FALSE.
    sim%pedestal_loc = 0.98343676_DP  - 0.04383409_DP
    sim%c_etb = 0.015_DP
    sim%c_crash = 0.5_DP
    sim%pressure_grad_threshold = 20.0_DP
    ! ------

    ALLOCATE(sim%prim%T(sim%nx+2*sim%nghosts), sim%prim%n(sim%nx+2*sim%nghosts))
    ALLOCATE(sim%grid%psin(sim%nx+2*sim%nghosts))
    ALLOCATE(sim%derived%p(sim%nx+2*sim%nghosts), sim%derived%alpha(sim%nx+2*sim%nghosts))
    ALLOCATE(sim%transparams%chi(sim%nx+2*sim%nghosts), sim%transparams%D(sim%nx+2*sim%nghosts), sim%transparams%V(sim%nx+2*sim%nghosts), sim%transparams%S_T(sim%nx+2*sim%nghosts), sim%transparams%S_N(sim%nx+2*sim%nghosts))
    ALLOCATE(canvas_data_x(sim%nx+2*sim%nghosts), canvas_data_y(sim%nx+2*sim%nghosts))
    ! Initialize arrays
    !
    ! Initialise x-domain
    sim%grid%psin = 0.0_DP
    CALL LINSPACE_RETSTEP(sim%grid%psin(1+sim%nghosts:sim%nx+sim%nghosts), 0.0_DP, 1.0_DP, sim%dx)
    ! sim%grid%psin(1+sim%nghosts+sim%nx:sim%nx+2*sim%nghosts) = 1.0_DP
    sim%grid%psin(sim%nx+1+sim%nghosts) = 1.0_DP + sim%dx
    sim%grid%psin(1) = 0.0_DP - sim%dx

    sim%prim%n = 0.0_DP
    sim%prim%T = 0.0_DP
    sim%transparams%S_T = 0.0_DP
    call LINEAR_GAUSSIAN(sim%grid%psin(1+sim%nghosts:sim%nx+sim%nghosts), sim%transparams%S_T(1+sim%nghosts:sim%nx+sim%nghosts), sim%power_input*0.01_DP, 0.4_DP, 0.2_DP, 0.8_DP)
    CALL MTANH(sim%grid%psin, sim%prim%T, 0.50173193_DP, 0.0_DP, 0.14852205_DP, 0.98343676_DP, 0.04383409_DP)

    CALL RSQUARED(sim%grid%psin, sim%prim%n, 5.0_DP)
    sim%prim%n(sim%nx + 2*sim%nghosts) = 0.0_DP
    ! print *, sim%prim%n(sim%nx - 1), sim%prim%n(sim%nx), sim%prim%n(sim%nx + 1), sim%prim%n(sim%nx + 2)

    sim%transparams%D   = 0.2_DP
    sim%transparams%V   = 0.0_DP
    sim%transparams%S_N = 0.0_DP
    sim%transparams%Chi = 0.5_DP
    !CALL RSQUARED(sim%grid%psin(1+sim%nghosts:sim%nx+sim%nghosts), sim%transparams%Chi(1+sim%nghosts:sim%nx+sim%nghosts), 5.0_DP)

    call update_transparams(sim)
    ! Call update transparams
    call update_ghosts_and_bcs(sim)
    call initialize_primtive_plots(temperature_plot, density_plot, transparam_plot, pressure_grad_plot)

    call init_window(ww, wh, "Fortran GOTY")
    ! call set_target_fps(1000)

    do while (.not. window_should_close())
        CALL computetimestep(sim, dt)
        tout = tout + dt
        t_lastelm = t_lastelm + dt
        call update_transparams(sim)
        call update(sim, dt)
        call update_ghosts_and_bcs(sim)
        ! INTER VS INTRA ELM,
        call mhd_stability_approximation(sim, t_lastelm)

        call begin_drawing()
        call clear_background(RAYWHITE)
        call plot_profile_given_subplot(temperature_plot, sim%grid%psin, SIM%PRIM%T, canvas_data_x, canvas_data_y, 0.0, 2.5, NOVABLACK)
        call plot_profile_given_subplot(density_plot, sim%grid%psin, SIM%PRIM%n, canvas_data_x, canvas_data_y,  0.0, 6.0, NOVARED)
        call plot_profile_given_subplot(transparam_plot, sim%grid%psin, SIM%transparams%chi, canvas_data_x, canvas_data_y,  0.0, 5.0, NOVABLACK)
        call plot_profile_given_subplot(transparam_plot, sim%grid%psin, SIM%transparams%D, canvas_data_x, canvas_data_y,  0.0, 5.0, NOVARED)
        call plot_profile_given_subplot(pressure_grad_plot, sim%grid%psin, SIM%derived%alpha, canvas_data_x, canvas_data_y,  0.0, 50.0, NOVAGREEN)
        ! BUFFER OVERFLOW HERE.... WHAT THE FUCK

        write (twritten, "(A4, F8.7, A1)") "t = ", tout
        int_buffer = measure_text(twritten, 20)
        call draw_text(twritten ,ww/2 - measure_text(trim(twritten), 20) / 2, 0, 20,BLACK)

        call draw_rectangle_lines_ex(temperature_plot, plot_thicc, BLACK)
        call draw_rectangle_lines_ex(density_plot, plot_thicc, BLACK)
        call draw_rectangle_lines_ex(transparam_plot, plot_thicc, BLACK)
        call draw_rectangle_lines_ex(pressure_grad_plot, plot_thicc, BLACK)

        call end_drawing()
    end do
END PROGRAM toy
