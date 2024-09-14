MODULE io
    use types_and_kinds
    use helpers
    IMPLICIT NONE
CONTAINS
    SUBROUTINE read_input_file(sim)
    ! Read input file and set parameters
    TYPE(Simulation), INTENT(INOUT) :: sim
    CHARACTER(LEN=100) :: line
    CHARACTER(LEN=100) :: varname
    CHARACTER(LEN=100) :: value
    INTEGER :: i
    INTEGER :: ierr
    ! Set default values
    sim%nx = 80
    sim%nghosts = 1
    sim%power_input = 10.5_DP
    sim%intra_elm_active = .FALSE.
    sim%pedestal_width = 0.04383409_DP
    sim%pedestal_loc = 0.98343676_DP  - sim%pedestal_width
    sim%c_etb = 0.5_DP
    sim%c_crash = 0.4_DP
    sim%pressure_grad_threshold = 250.0_DP
    sim%chi_0 = 1.5_DP
    sim%tgrad_crit = 4.0_DP

    ! Open input file
    OPEN(UNIT=10, FILE='input.txt', STATUS='OLD', ACTION='READ', IOSTAT=ierr)
    IF (ierr /= 0) THEN
        PRINT *, 'Error opening input file'
        STOP
    END IF
    ! Read input file
    ! Loop over lines
    DO
        READ(10, '(A)', IOSTAT=ierr) line
        IF (ierr /= 0) EXIT
        ! Skip comments
        IF (line(1:1) == '!') CYCLE
        ! Split line into variable name and value
        i = INDEX(line, '=')
        varname = TRIM(line(1:i-1))
        value = TRIM(line(i+1:))
        ! Set variable
        print *, varname, value
        SELECT CASE (varname)
        CASE ('nx')
            READ(value, *) sim%nx
        CASE ('nghosts')
            READ(value, *) sim%nghosts
        CASE ('power_input')
            READ(value, *) sim%power_input
        CASE ('intra_elm_active')
            READ(value, *) sim%intra_elm_active
        CASE ('pedestal_width')
            READ(value, *) sim%pedestal_width
        CASE ('pedestal_loc')
            READ(value, *) sim%pedestal_loc
        CASE ('c_etb')
            READ(value, *) sim%c_etb
        CASE ('c_crash')
            READ(value, *) sim%c_crash
        CASE ('pressure_grad_threshold')
            READ(value, *) sim%pressure_grad_threshold
        CASE ('chi_0')
            READ(value, *) sim%chi_0
        CASE ('tgrad_crit')
            READ(value, *) sim%tgrad_crit
        CASE DEFAULT
            PRINT *, 'Unknown variable name: ', varname
        END SELECT
    END DO
    END SUBROUTINE read_input_file
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
END MODULE IO
