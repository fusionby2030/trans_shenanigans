FC=gfortran
FFLAGS=-std=f2003  -ggdb -g3 -Wall -Wextra -Wno-conversion -ffree-line-length-none
LIBS= -lraylib
OPT=-g3 -ggdb
BIN=test

all: default

raymath.o:raymath.f03
	${FC} ${OPT} -c ${FFLAGS} raymath.f03

raylib.o:raymath.o
	${FC} ${OPT} -c ${FFLAGS} raylib.f03

program.o:raylib.o main.f03
	${FC} ${OPT} -c ${FFLAGS} main.f03
	${FC} ${OPT} -o  ${BIN} main.o raylib.o raymath.o ${LIBS}

default: program.o

allclean:
	rm main.o
	rm raymath.o
	rm raylib.o
	rm ${BIN}

clean:
	rm main.o
	rm raymath.o
	rm raylib.o
