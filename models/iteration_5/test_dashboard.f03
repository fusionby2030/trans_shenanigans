MODULE PLOTTING 
    use types_and_kinds
    use iso_c_binding, only: c_int, c_int32_t, C_NULL_CHAR, C_NULL_PTR, c_loc, c_float
    use raylib
    use raymath
    IMPLICIT NONE
    ! 1780
    INTEGER(c_int), PARAMETER :: ww = 1800, wh = 900, border_padding=20
    INTEGER(c_int), PARAMETER :: plot_width=0.9*(ww/3), plot_height=0.9*(wh/3)
    REAL(C_FLOAT), PARAMETER ::  plot_thicc=3.0
    REAL(C_FLOAT), PARAMETER :: dot_radius=5.5
    INTEGER(c_int), PARAMETER :: axis_fontsize=20
CONTAINS 
    SUBROUTINE initialize_plots(temp_plot, chi_plot, power_plot, chi_crash_plot)
        TYPE(RECTANGLE), INTENT(INOUT) :: temp_plot, chi_plot, power_plot, chi_crash_plot
        ! Top Left
        temp_plot%x      = border_padding
        temp_plot%y      = border_padding
        temp_plot%width  = plot_width
        temp_plot%height = plot_height
        ! Middle left 
        chi_plot%x      = border_padding
        chi_plot%y      = wh / 2 - plot_height / 2
        chi_plot%width  = plot_width
        chi_plot%height = plot_height
        ! Bottom Left 
        power_plot%x      = border_padding
        power_plot%y      = wh - plot_height - border_padding
        power_plot%width  = plot_width
        power_plot%height = plot_height
        ! Middle Middle 
        chi_crash_plot%x      = ww / 2 - plot_width / 2
        chi_crash_plot%y      = wh / 2 - plot_height / 2
        chi_crash_plot%width  = plot_width
        chi_crash_plot%height = plot_height
    END SUBROUTINE initialize_plots
    SUBROUTINE plot_profile_given_subplot(rect, x, y, y_l_lim, y_u_lim, color, title)
        TYPE(RECTANGLE), INTENT(IN) :: rect
        REAL(DP), INTENT(IN) :: x(:), y(:)
        REAL, INTENT(IN) :: y_l_lim, y_u_lim
        CHARACTER(len=6) :: charbuff
        CHARACTER(LEN=50), INTENT(IN), OPTIONAL :: title
        INTEGER(c_int32_t), INTENT(IN) :: color
        REAL(DP) :: buff_x(size(x)), buff_y(size(y))
        integer(IP) :: i
        REAL(DP) :: max_x, max_y, min_x, min_y

        max_x = maxval(x) 
        min_x = minval(x) 
        max_y = real(y_u_lim) 
        min_y = real(y_l_lim) 
        buff_x = (max_x - x) / (max_x - min_x )
        buff_y = (max_y - y) / (max_y - min_y)
        buff_x = 1-buff_x
        buff_x = buff_x * rect%width
        buff_y = buff_y * rect%height

        do i=2, size(x)-1
            CALL draw_circle(int(buff_x(i) + rect%x), int(buff_y(i)+ rect%y), dot_radius, COLOR)
        end do

        write (charbuff, '(F6.2)') min_x
        call draw_text(charbuff//C_NULL_CHAR, int(rect%x - AXIS_FONTSIZE), int(rect%y + rect%height + AXIS_FONTSIZE / 4), AXIS_FONTSIZE, BLACK)
        write (charbuff, '(F6.2)') max_x
        call draw_text(charbuff//C_NULL_CHAR, int(rect%x + rect%width - 1.5*AXIS_FONTSIZE ), int(rect%y + rect%height + AXIS_FONTSIZE / 4), AXIS_FONTSIZE, BLACK)
        write (charbuff, '(F6.2)') max_y
        call draw_text(charbuff//C_NULL_CHAR, int(rect%x - AXIS_FONTSIZE / 2), int(rect%y + AXIS_FONTSIZE / 2), AXIS_FONTSIZE, BLACK)

        call draw_text(title//C_NULL_CHAR, int(rect%x + rect%width / 2 - 6*AXIS_FONTSIZE), int(rect%y - AXIS_FONTSIZE), AXIS_FONTSIZE, BLACK)
    END SUBROUTINE plot_profile_given_subplot
END MODULE PLOTTING 
MODULE TESTS 
    use types_and_kinds
    use helpers
    use io 
    use physics 
    use plotting 
    IMPLICIT NONE
CONTAINS
    SUBROUTINE ETB_CHI(sim, rect)
        TYPE(SIMULATION), INTENT(IN) :: sim 
        TYPE(SIMULATION) :: simcopy
        TYPE(RECTANGLE), INTENT(IN) :: rect
        INTEGER :: I 
        REAL(DP), DIMENSION(8) :: c_etb_values!  = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.5]
        INTEGER, DIMENSION(8) :: colors = [BLACK, NOVARED, NOVABLUE, NOVAGREEN, RED, GREEN, BLUE, BLACK]
        simcopy = sim 
        call LINSPACE(c_etb_values, 0.0_DP, 0.01_DP)
        
        DO i = 1, SIZE(c_etb_values)
            simcopy%c_etb = c_etb_values(i)
            call update_transparams(simcopy)
            call plot_profile_given_subplot(rect, simcopy%grid%psin, simcopy%transparams%chi, 0.0, 4.0, colors(i), 'INTER-ELM Diffusivity (m^2/s)')
        END DO
    END SUBROUTINE ETB_CHI
    SUBROUTINE POWER_SCAN(sim, rect)
        TYPE(SIMULATION), INTENT(IN) :: sim 
        TYPE(SIMULATION) :: simcopy
        TYPE(RECTANGLE), INTENT(IN) :: rect
        INTEGER :: I 
        REAL(DP), DIMENSION(8) :: scan_values  ! [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.5]
        INTEGER, DIMENSION(8) :: colors = [BLACK, NOVARED, NOVABLUE, NOVAGREEN, RED, GREEN, BLUE, BLACK]

        call LINSPACE(scan_values, 1.0_DP, 20.0_DP)
        simcopy = sim
        DO i = 1, SIZE(scan_values)
            simcopy%power_input = scan_values(i)
            call LINEAR_GAUSSIAN(simcopy%grid%psin(1+simcopy%nghosts:simcopy%nx+simcopy%nghosts), simcopy%transparams%S_T(1+simcopy%nghosts:simcopy%nx+simcopy%nghosts), simcopy%power_input*0.01_DP, 0.4_DP, 0.2_DP, 0.8_DP)
            call plot_profile_given_subplot(rect, simcopy%grid%psin, simcopy%transparams%S_T, 0.0, 0.05, colors(i), 'Power Scan (MW/m^2)'//C_NULL_CHAR)
        END DO
    END SUBROUTINE POWER_SCAN
    SUBROUTINE CRASH_CHI(sim, rect)
        TYPE(SIMULATION), INTENT(IN) :: sim 
        TYPE(SIMULATION) :: simcopy
        TYPE(RECTANGLE), INTENT(IN) :: rect
        INTEGER :: I 
        REAL(DP), DIMENSION(8) :: crash_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.5]
        INTEGER, DIMENSION(8)  :: colors = [BLACK, NOVARED, NOVABLUE, NOVAGREEN, RED, GREEN, BLUE, BLACK]
        simcopy = sim
        simcopy%intra_elm_active = .TRUE.
        DO i = 1, SIZE(crash_values)
            simcopy%c_crash = crash_values(i)
            call update_transparams(simcopy)
            call plot_profile_given_subplot(rect, simcopy%grid%psin, simcopy%transparams%chi, 0.0, 4.0, colors(i), 'INTRA-ELM Diffusivity (m^2/s)')
        END DO
        simcopy%intra_elm_active = .FALSE.
    END SUBROUTINE CRASH_CHI
END MODULE TESTS

PROGRAM test_dashboard 
    use types_and_kinds
    use helpers 
    use io 
    use physics 
    use iso_c_binding, only: c_int, c_int32_t, C_NULL_CHAR, C_NULL_PTR, c_loc, c_float
    use raylib
    use raymath
    use plotting
    use tests 
IMPLICIT NONE 
    TYPE(SIMULATION) :: sim 
    TYPE(RECTANGLE) :: temp_plot, chi_etb_plot, chi_crash_plot, power_plot
    call read_input_file(sim)
    CALL initialize_simulation(sim)

    call update_transparams(sim)

    call initialize_plots(temp_plot, chi_etb_plot, power_plot, chi_crash_plot)
    call init_window(ww, wh, ""//C_NULL_CHAR)
    DO while (.not. window_should_close())
        call begin_drawing()
        call clear_background(RAYWHITE)

        call plot_profile_given_subplot(temp_plot, sim%grid%psin, sim%prim%T, 0.0, 2.5, BLACK, 'Temperature (keV)')
        
        CALL ETB_CHI(sim, chi_etb_plot)
        CALL POWER_SCAN(sim, power_plot)
        CALL CRASH_CHI(sim, chi_crash_plot)
        
        call draw_rectangle_lines_ex(temp_plot, plot_thicc, BLACK)
        call draw_rectangle_lines_ex(chi_etb_plot, plot_thicc, BLACK)
        call draw_rectangle_lines_ex(power_plot, plot_thicc, BLACK)
        call draw_rectangle_lines_ex(chi_crash_plot, plot_thicc, BLACK)
        call end_drawing()
    END DO 
    ! DEALLOCATE(canvas_data_x, canvas_data_y)
    
END PROGRAM test_dashboard