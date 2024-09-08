PROGRAM sankt_sven
    use types_and_kinds
    use physics
    use helpers
IMPLICIT NONE
    TYPE(Simulation) :: sim
    REAL(DP) :: tout=0.0, dt=0.01, wout=0.0, wstep=0.00005, totalsimtime=1.0_dp
    REAL(DP) :: t_lastelm=0.0_DP ! 200 microseconds -> 0.0002

    ! ---- Setup ---
    sim%nx = 80
    sim%nghosts = 1
    sim%power_input = 10.5_DP
    sim%intra_elm_active = .FALSE.
    sim%pedestal_loc = 0.98343676_DP  - 0.04383409_DP
    sim%c_etb = 0.0145_DP
    sim%c_crash = 0.5_DP
    sim%pressure_grad_threshold = 225.0_DP
    sim%chi_0 = 1.5_DP
    ! ------

    ALLOCATE(sim%prim%T(sim%nx+2*sim%nghosts), sim%prim%n(sim%nx+2*sim%nghosts))
    ALLOCATE(sim%grid%psin(sim%nx+2*sim%nghosts))
    ALLOCATE(sim%derived%p(sim%nx+2*sim%nghosts), sim%derived%alpha(sim%nx+2*sim%nghosts))
    ALLOCATE(sim%transparams%chi(sim%nx+2*sim%nghosts), sim%transparams%D(sim%nx+2*sim%nghosts), sim%transparams%V(sim%nx+2*sim%nghosts), sim%transparams%S_T(sim%nx+2*sim%nghosts), sim%transparams%S_N(sim%nx+2*sim%nghosts))

    sim%grid%psin = 0.0_DP
    ! CALL LINSPACE_RETSTEP(sim%grid%psin(1+sim%nghosts:sim%nx+sim%nghosts), 0.0_DP, 1.0_DP, sim%dx)
    CALL LINSPACE_RETSTEP(sim%grid%psin, 0.8_DP, 1.0_DP, sim%dx)
    ! sim%grid%psin(1+sim%nghosts+sim%nx:sim%nx+2*sim%nghosts) = 1.0_DP
    ! sim%grid%psin(sim%nx+1+sim%nghosts) = 1.0_DP + sim%dx
    ! sim%grid%psin(1) = 0.0_DP - sim%dx

    sim%pedestal_idx = 1+sim%nghosts+int(sim%pedestal_loc/sim%dx)
    print *, "Pedestal index: ", sim%pedestal_idx
    sim%prim%n = 0.0_DP
    sim%prim%T = 0.0_DP
    sim%transparams%S_T = 0.0_DP
    call LINEAR_GAUSSIAN(sim%grid%psin(1+sim%nghosts:sim%nx+sim%nghosts), sim%transparams%S_T(1+sim%nghosts:sim%nx+sim%nghosts), sim%power_input*0.01_DP, 0.4_DP, 0.2_DP, 0.8_DP)
    CALL MTANH(sim%grid%psin, sim%prim%T, 0.50173193_DP, 0.0_DP, 0.14852205_DP, 0.98343676_DP, 0.04383409_DP)

    ! CALL RSQUARED(sim%grid%psin, sim%prim%n, 5.0_DP)
    call MTANH(sim%grid%psin, sim%prim%n, 3.44198_DP, 0.0_DP, 0.030139_DP, 0.99695_DP, 0.05961_DP)
    ! "prof_height, prof_slope, prof_position, prof_width = 3.44198, 0.030139,0.99695,0.05961
    !
    sim%prim%n(sim%nx + 2*sim%nghosts) = 0.0_DP
    ! print *, sim%prim%n(sim%nx - 1), sim%prim%n(sim%nx), sim%prim%n(sim%nx + 1), sim%prim%n(sim%nx + 2)

    sim%transparams%D   = sim%chi_0
    sim%transparams%V   = 0.0_DP
    sim%transparams%S_N = 20.0_DP
    sim%transparams%Chi = 0.5_DP

    call update_transparams(sim)
    ! Call update transparams
    call update_ghosts_and_bcs(sim)

    DO
        CALL computetimestep(sim, dt)
        WRITE (*, '(A, 1X, F8.7, 1X, A, 1X, F8.4)') 'dt    ', dt, 'simtime    ', tout
        tout = tout + dt
        t_lastelm = t_lastelm + dt
        call update_transparams(sim)
        call update(sim, dt)
        call update_ghosts_and_bcs(sim)
        ! INTER VS INTRA ELM,
        call mhd_stability_approximation(sim, t_lastelm)

        IF (tout > totalsimtime) EXIT
    END DO
END PROGRAM sankt_sven
