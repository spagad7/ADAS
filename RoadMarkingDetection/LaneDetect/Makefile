################################################################################
# Trigger Makefile
################################################################################

################################################################################
# Key paths and settings
################################################################################
CFLAGS += -std=c++11
CVFLAGS = `pkg-config --cflags opencv`
CC = g++ -fopenmp ${CFLAGS} -ggdb ${CVFLAGS}
OUTPUTNAME = laneDetect${D}

OUTDIR = bin/

################################################################################
# Dependencies
################################################################################
CV_LIB = `pkg-config --libs opencv`${D}

################################################################################
# Master inc/lib/obj/dep settings
################################################################################
OBJ = laneDetect.o
LIB += ${CV_LIB}

################################################################################
# Rules/recipes
################################################################################
# Final binary
${OUTPUTNAME}: ${OBJ}
	${CC} -o ${OUTPUTNAME} ${OBJ} ${LIB}
	mv ${OUTPUTNAME} ${OUTDIR}

# Intermediate objects
%.o: %.cpp
	${CC} ${CFLAGS} ${INC} -Wall -c -D LINUX $*.cpp

# Clean up intermediate objects
clean_obj:
	rm -f ${OBJ}	@echo "all cleaned up!"

# Clean up everything.
clean:
	rm -f ${OUTDIR}/${OUTPUTNAME} ${OBJ}	@echo "all cleaned up!"
