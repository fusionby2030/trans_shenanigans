MODULE types_and_kinds
    IMPLICIT NONE
    INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(15, 307)
    INTEGER, PARAMETER :: ip = SELECTED_INT_KIND(8)
    REAL(DP), PARAMETER :: PI = 4.0_DP * ATAN(1.0_DP)
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
        INTEGER(IP) :: nx, nghosts, pedestal_idx
        REAL(DP) :: chi_0, critical_grad, dx
        REAL(DP) :: power_input
        REAL(DP) :: c_etb, c_crash
        REAL(DP) :: pedestal_loc, alpha_critical, pressure_grad_threshold, pedestal_width
        LOGICAL :: intra_elm_active
        type(primitives) :: prim
        type(coeffs)     :: transparams
        type(domain)     :: grid
        type(derived)    :: derived
    END TYPE simulation
END MODULE types_and_kinds
MODULE helpers
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
END MODULE helpers
