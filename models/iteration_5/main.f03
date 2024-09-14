PROGRAM sankt_sven
    use types_and_kinds
    use physics
    use helpers
    use io
IMPLICIT NONE
    TYPE(Simulation) :: sim
    REAL(DP) :: tout=0.0, dt=0.01, wout=0.0, wstep=0.0001, totalsimtime=1.0_dp
    REAL(DP) :: t_lastelm=0.0_DP ! 200 microseconds -> 0.0002

    call read_input_file(sim)
    CALL initialize_simulation(sim)

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
END PROGRAM sankt_sven
