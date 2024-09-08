# make dashboard.o for RAYLIB shenanigans or make for simple
FC=gfortran
FFLAGS=-std=f2003 -Wall -Wextra -Wno-conversion -ffree-line-length-none
OPTMODEL=-O2
OPTGUI=-g3 -ggdb
BIN=main
BINGUI=gui
LIBS= -lraylib
all: default

helpers.o:helpers.f03
	${FC} ${OPT} ${FOPT} -c ${FFLAGS} helpers.f03

physics.o:helpers.o physics.f03
	${FC} ${OPT} ${FOPT} -c ${FFLAGS} physics.f03

program.o:physics.o main.f03
	${FC} ${OPT} ${FOPT} -c ${FFLAGS} main.f03
	${FC} ${OPT} ${FOPT} -fsanitize=address -o  ${BIN} main.o helpers.o physics.o

raymath.o:raymath.f03
	${FC} ${OPTGUI} -c ${FFLAGS} raymath.f03

raylib.o:raymath.o
	${FC} ${OPTGUI} -c ${FFLAGS} raylib.f03

dashboard.o:raylib.o physics.o dashboard.f03
	${FC} ${OPTGUI} -c ${FFLAGS} dashboard.f03
	${FC} ${OPTGUI} -fsanitize=address -o  ${BINGUI} dashboard.o helpers.o physics.o raylib.o raymath.o ${LIBS}

default: program.o

allclean:
	rm *.o
	rm *.mod
	rm ${BIN}
	rm ${BINGUI}

clean:
	rm *.mod
	rm *.o
	rm ${BIN}
	rm ${BINGUI}
