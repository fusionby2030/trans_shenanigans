MODULE plotting_helpers
    use types_and_kinds
    use iso_c_binding, only: c_int, c_int32_t, C_NULL_CHAR, C_NULL_PTR, c_loc, c_float
    use raylib
    use raymath
    IMPLICIT NONE
    ! 1780
    INTEGER(c_int), PARAMETER :: ww = 1800, wh = 900, border_padding=20
    INTEGER(c_int), PARAMETER :: plot_width=0.9*(ww/3), plot_height=0.9*(wh/3)
    REAL(c_float), PARAMETER ::  plot_thicc=3.0
    REAL(C_FLOAT), PARAMETER :: dot_radius=5.5
CONTAINS
    SUBROUTINE initialize_primtive_plots(tp, np, coefp, pp, pedtt)
        TYPE(Rectangle), INTENT(INOUT)  :: tp, np, pp, coefp, pedtt
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
        ! Bottom Left, half plot height
        pedtt%x      = border_padding
        pedtt%y      = wh - plot_height - border_padding
        pedtt%width  = plot_width
        pedtt%height = plot_height / 2
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

        max_x = maxval(x) ! x_u_lim ! maxval(x)
        min_x = minval(x) ! minval(x)
        max_y = real(y_u_lim) ! 2.5 ! maxval(y)
        min_y = real(y_l_lim) ! 0.0 !minval(y)
        buff_x = (max_x - x) / (max_x - min_x )
        buff_y = (max_y - y) / (max_y - min_y)
        buff_x = 1-buff_x
        buff_x = buff_x * rect%width
        buff_y = buff_y * rect%height

        do i=2, size(x)-1
            CALL draw_circle(int(buff_x(i) + rect%x), int(buff_y(i)+ rect%y), dot_radius, COLOR)
        end do
        ! call draw_circle(int(buff_x(1) + rect%x), int(buff_y(1)+ rect%y), 2.0, BLUE)
        ! call draw_circle(int(buff_x(size(x)) + rect%x), int(buff_y(size(x))+ rect%y), 2.0, BLUE)

        write (charbuff, '(F6.2)') min_x
        call draw_text(charbuff//C_NULL_CHAR, int(rect%x), int(rect%y + rect%height), 20, BLACK)
        write (charbuff, '(F6.2)') max_x
        call draw_text(charbuff//C_NULL_CHAR, int(rect%x + rect%width), int(rect%y + rect%height), 20, BLACK)
        write (charbuff, '(F6.2)') max_y
        call draw_text(charbuff//C_NULL_CHAR, int(rect%x), int(rect%y), 20, BLACK)
    END SUBROUTINE plot_profile_given_subplot
END MODULE plotting_helpers

PROGRAM toy
    use types_and_kinds
    use physics
    use helpers
    use iso_c_binding, only: c_int, c_int32_t, C_NULL_CHAR, C_NULL_PTR, c_loc, c_float
    use raylib
    use raymath
    use plotting_helpers
    IMPLICIT NONE
    TYPE(Simulation) :: sim
    TYPE(Rectangle)  :: temperature_plot, density_plot, pressure_grad_plot, transparam_plot, pedtimetrace_plot
    character(len=12) :: twritten
    REAL(DP) :: tout=0.0, dt=0.01, wout=0.0, wstep=0.00005, totalsimtime=1.0_dp
    REAL(DP) :: t_lastelm=0.0_DP ! 200 microseconds -> 0.0002
    REAL(DP) :: plot_arrays(4, 1000)
    INTEGER(IP) :: plot_index = 1
    type(Vector2) :: vec_buffer
    REAL(DP), ALLOCATABLE :: canvas_data_x(:), canvas_data_y(:)
    INTEGER :: int_buffer
    ! ---- Setup ---
    sim%nx = 80
    sim%nghosts = 1
    sim%power_input = 10.5_DP
    sim%intra_elm_active = .FALSE.
    sim%pedestal_width = 0.04383409_DP
    sim%pedestal_loc = 0.98343676_DP  - sim%pedestal_width
    sim%c_etb = 0.0135_DP
    sim%c_crash = 0.46_DP
    sim%pressure_grad_threshold = 220.0_DP
    sim%chi_0 = 1.5_DP
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

    sim%transparams%D   = 0.0_DP ! sim%chi_0 ! 1.2_DP

    sim%transparams%V   = 0.0_DP
    sim%transparams%S_N = 0.0_DP
    sim%transparams%Chi = 0.5_DP
    !CALL RSQUARED(sim%grid%psin(1+sim%nghosts:sim%nx+sim%nghosts), sim%transparams%Chi(1+sim%nghosts:sim%nx+sim%nghosts), 5.0_DP)

    call update_transparams(sim)
    ! Call update transparams
    call update_ghosts_and_bcs(sim)
    call initialize_primtive_plots(temperature_plot, density_plot, transparam_plot, pressure_grad_plot, pedtimetrace_plot)

    call init_window(ww, wh, "Fortran GOTY")
    call set_target_fps(60)
    plot_arrays = 0.0_DP
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
        ! Store teped and neped every 1/(size(plot_arrays)) seconds
        if (tout - real(plot_index) * 1.0_DP / real(size(plot_arrays, 2)) > 0.0_DP) then
            plot_arrays(1, plot_index) = sim%prim%T(sim%pedestal_idx)
            plot_arrays(2, plot_index) = sim%prim%n(sim%pedestal_idx)
            plot_arrays(3, plot_index) = tout
            ! plot_arrays(4, plot_index) = sim%derived%alpha(sim%nx)
            plot_index = plot_index + 1
            print *, plot_index
        end if
        call clear_background(RAYWHITE)
        call plot_profile_given_subplot(temperature_plot, sim%grid%psin, SIM%PRIM%T, canvas_data_x, canvas_data_y, 0.0, 1.0, NOVABLACK)
        call plot_profile_given_subplot(density_plot, sim%grid%psin, SIM%PRIM%n, canvas_data_x, canvas_data_y,  0.0, 6.0, NOVARED)
        call plot_profile_given_subplot(transparam_plot, sim%grid%psin, SIM%transparams%chi, canvas_data_x, canvas_data_y,  0.0, 5.0, NOVABLACK)
        call plot_profile_given_subplot(transparam_plot, sim%grid%psin, SIM%transparams%D, canvas_data_x, canvas_data_y,  0.0, 5.0, NOVARED)
        call plot_profile_given_subplot(pressure_grad_plot, sim%grid%psin, SIM%derived%alpha, canvas_data_x, canvas_data_y,  0.0, real(sim%pressure_grad_threshold), NOVAGREEN)
        ! call plot_profile_given_subplot(pedtimetrace_plot, plot_arrays(3, :), plot_arrays(1, :), canvas_data_x, canvas_data_y,  0.0, 1.5, NOVABLACK)
        ! BUFFER OVERFLOW HERE.... WHAT THE FUCK

        write (twritten, "(A4, F8.7, A1)") "t = ", tout
        int_buffer = measure_text(twritten, 20)
        call draw_text(twritten ,ww/2 - measure_text(trim(twritten), 20) / 2, 0, 20,BLACK)
        if (sim%intra_elm_active .eqv. .True.) then
            call draw_text('ELM', ww/2, wh/2, 30, RED)
        end if
        call draw_rectangle_lines_ex(temperature_plot, plot_thicc, BLACK)
        call draw_rectangle_lines_ex(density_plot, plot_thicc, BLACK)
        call draw_rectangle_lines_ex(transparam_plot, plot_thicc, BLACK)
        call draw_rectangle_lines_ex(pressure_grad_plot, plot_thicc, BLACK)
        call draw_rectangle_lines_ex(pedtimetrace_plot, plot_thicc, BLACK)

        call end_drawing()
    end do
END PROGRAM toy
