PROGRAM sankt_sven
    use types_and_kinds
    use physics
    use helpers
IMPLICIT NONE
    TYPE(Simulation) :: sim
    REAL(DP) :: tout=0.0, dt=0.01, wout=0.0, wstep=0.0001, totalsimtime=1.0_dp
    REAL(DP) :: t_lastelm=0.0_DP ! 200 microseconds -> 0.0002

    ! ---- Setup ---
    sim%nx = 80
    sim%nghosts = 1
    sim%power_input = 10.5_DP
    sim%intra_elm_active = .FALSE.
    sim%pedestal_width = 0.04383409_DP
    sim%pedestal_loc = 0.98343676_DP  - sim%pedestal_width
    sim%c_etb = 0.014_DP
    sim%c_crash = 0.4_DP
    sim%pressure_grad_threshold = 250.0_DP
    sim%chi_0 = 1.5_DP
    ! ------

    ALLOCATE(sim%prim%T(sim%nx+2*sim%nghosts), sim%prim%n(sim%nx+2*sim%nghosts))
    ALLOCATE(sim%grid%psin(sim%nx+2*sim%nghosts))
    ALLOCATE(sim%derived%p(sim%nx+2*sim%nghosts), sim%derived%alpha(sim%nx+2*sim%nghosts))
    ALLOCATE(sim%transparams%chi(sim%nx+2*sim%nghosts), sim%transparams%D(sim%nx+2*sim%nghosts), sim%transparams%V(sim%nx+2*sim%nghosts), sim%transparams%S_T(sim%nx+2*sim%nghosts), sim%transparams%S_N(sim%nx+2*sim%nghosts))

    sim%grid%psin = 0.0_DP
    CALL LINSPACE_RETSTEP(sim%grid%psin, 0.8_DP, 1.0_DP, sim%dx)

    sim%pedestal_idx = 1+sim%nghosts+int(sim%pedestal_loc/sim%dx)

    call LINEAR_GAUSSIAN(sim%grid%psin(1+sim%nghosts:sim%nx+sim%nghosts), sim%transparams%S_T(1+sim%nghosts:sim%nx+sim%nghosts), sim%power_input*0.01_DP, 0.4_DP, 0.2_DP, 0.8_DP)
    CALL MTANH(sim%grid%psin, sim%prim%T, 0.50173193_DP, 0.0_DP, 0.14852205_DP, 0.98343676_DP, 0.04383409_DP)
    call MTANH(sim%grid%psin, sim%prim%n, 3.44198_DP, 0.0_DP, 0.030139_DP, 0.99695_DP, 0.05961_DP)
    ! call MTANH(sim%grid%psin, sim%prim%n, 3.44198_DP, 0.0_DP, 0.030139_DP, sim%pedestal_loc, sim%pedestal_width)

    ! CALL RSQUARED(sim%grid%psin, sim%prim%n, 4.0_DP)
    ! "prof_height, prof_slope, prof_position, prof_width = 3.44198, 0.030139,0.99695,0.05961
    sim%prim%n(sim%nx + 2*sim%nghosts) = 0.0_DP
    sim%transparams%D   = 0.0_DP ! sim%chi_0
    sim%transparams%V   = 0.0_DP
    sim%transparams%S_N = 0.0_DP ! 20.0_DP

    sim%transparams%Chi = 0.5_DP
    call update_transparams(sim)
    call update_ghosts_and_bcs(sim)
OPEN(UNIT=10, FILE='./output.txt', STATUS='REPLACE')
    CALL WRITE_HEADER(sim, 10)
    CALL WRITESTATE(sim, tout, 10)
    DO
        CALL computetimestep(sim, dt)
        WRITE (*, '(A, 1X, F8.7, 1X, A, 1X, F8.4)') 'dt    ', dt, 'simtime    ', tout
        tout = tout + dt
        t_lastelm = t_lastelm + dt
        wout = wout + dt
        call update_transparams(sim)
        call update(sim, dt)
        call update_ghosts_and_bcs(sim)
        ! INTER VS INTRA ELM,
        call mhd_stability_approximation(sim, t_lastelm)

        IF (tout >= totalsimtime) EXIT
        ! .OR. (sim%intra_elm_active .eqv. .True.))
        IF (wout >= wstep) THEN
            CALL WRITESTATE(sim, tout, 10)
            wout = 0.0_DP
        END IF

    END DO
    close(10)
CONTAINS
    SUBROUTINE WRITE_HEADER(SIM, funit)
        TYPE(Simulation), INTENT(IN) :: sim
        INTEGER :: funit
        WRITE (funit, '(A, 1X, I4)' ) 'nx    ', sim%nx
        WRITE (funit, '(A, 1X, I4)' ) 'nghost', sim%nghosts
        WRITE (funit, '(A, 1X, F8.4)' ) 'dx    ', sim%dx
        WRITE (funit, '(A, 1X, *(F8.4, 1X))') 'psin', sim%grid%psin
        ! sim%power_input = 10.5_DP
        ! sim%intra_elm_active = .FALSE.
        ! sim%pedestal_loc = 0.98343676_DP  - 0.04383409_DP
        ! sim%c_etb = 0.0145_DP
        ! sim%c_crash = 0.5_DP
        ! sim%pressure_grad_threshold = 225.0_DP
        ! sim%chi_0 = 1.5_DP'
        WRITE (funit, '(A, 1X, F8.4)' ) 'power_input', sim%power_input
        WRITE (funit, '(A, 1X, F8.4)' ) 'pedestal_loc', sim%pedestal_loc
        WRITE (funit, '(A, 1X, F8.4)' ) 'c_etb', sim%c_etb
        WRITE (funit, '(A, 1X, F8.4)' ) 'c_crash', sim%c_crash
        WRITE (funit, '(A, 1X, F8.4)' ) 'pressure_grad_threshold', sim%pressure_grad_threshold
        WRITE (funit, '(A, 1X, F8.4)' ) 'chi_0', sim%chi_0

    END SUBROUTINE WRITE_HEADER
    SUBROUTINE WRITESTATE(sim, tout, funit)
        IMPLICIT NONE
        TYPE(Simulation), INTENT(IN) :: sim
        REAL(DP), INTENT(IN) :: tout
        INTEGER :: funit
        WRITE (funit, '(A, 1X, F8.4)') 'tout  ', tout
        WRITE (funit, '(A, 1X, L1)') 'mode ', sim%intra_elm_active
        WRITE (funit, '(A, 1X, *(F8.4, 1X))') 'temperature', sim%prim%T
        WRITE (funit, '(A, 1X, *(F8.4, 1X))') 'density', sim%prim%n
        WRITE (funit, '(A, 1X, *(F8.4, 1X))') 'pressure', sim%derived%p
        WRITE (funit, '(A, 1X, *(F8.4, 1X))') 'alpha', sim%derived%alpha
        WRITE (funit, '(A, 1X, *(F8.4, 1X))') 'trans_chi', sim%transparams%chi
        WRITE (funit, '(A, 1X, *(F8.4, 1X))') 'trans_D', sim%transparams%D
        WRITE (funit, '(A, 1X, *(F8.4, 1X))') 'trans_V', sim%transparams%V
    END SUBROUTINE WRITESTATE

END PROGRAM sankt_sven
