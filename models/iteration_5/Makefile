# make dashboard.o for RAYLIB shenanigans or make for simple
FC=gfortran
FFLAGS=-std=f2003 -Wall -Wextra -Wno-conversion -ffree-line-length-none
FOPT=-O2
OPTGUI=-g3 -ggdb
BIN=main
BINGUI=gui
LIBS= -lraylib
all: default

helpers.o:helpers.f03
	${FC} ${OPT} ${FOPT} -c ${FFLAGS} helpers.f03

physics.o:helpers.o physics.f03
	${FC} ${OPT} ${FOPT} -c ${FFLAGS} physics.f03

io.o:physics.o io.f03
	${FC} ${OPT} ${FOPT} -c ${FFLAGS} io.f03

# -fsanitize=address
program.o:io.o main.f03
	${FC} ${OPT} ${FOPT} -c ${FFLAGS} main.f03
	${FC} ${OPT} ${FOPT} -o  ${BIN} main.o helpers.o physics.o io.o

raymath.o:raymath.f03
	${FC} ${OPTGUI} -c ${FFLAGS} raymath.f03

raylib.o:raymath.o
	${FC} ${OPTGUI} -c ${FFLAGS} raylib.f03

#
dashboard.o:raylib.o io.o physics.o dashboard.f03
	${FC} ${OPTGUI} -c ${FFLAGS} dashboard.f03
	${FC} ${OPTGUI} -o ${BINGUI} dashboard.o helpers.o physics.o io.o raylib.o raymath.o ${LIBS}

test.o:raylib.o io.o test_dashboard.f03
	${FC} ${OPTGUI} -c ${FFLAGS} test_dashboard.f03
	${FC} ${OPTGUI} -o  test_dashboard test_dashboard.o helpers.o physics.o io.o raylib.o raymath.o ${LIBS}

default: program.o

allclean:
	rm *.o
	rm *.mod
	rm ${BIN}
	rm output.txt
clean:
	rm *.mod
	rm *.o
	rm ${BIN}
	rm output.txt
	rm simulation_state.pkl
