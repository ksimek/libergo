INCLUDE_PATHS:=-I../ -I/usr/local/include
CXX_FLAGS := -g
RM := rm -rf

all: mh hmc hmc_vec record

mh: mh.o
	g++ mh.o -o mh
mh.o: mh.cpp ../ergo/mh.h ../ergo/util.h ../ergo/rand.h
	g++ -c mh.cpp ${CXX_FLAGS} ${INCLUDE_PATHS} -o mh.o

hmc: hmc.o
	g++ hmc.o -o hmc
hmc.o: hmc.cpp ../ergo/hmc.h ../ergo/util.h ../ergo/rand.h
	g++ -c hmc.cpp ${CXX_FLAGS} ${INCLUDE_PATHS} -o hmc.o

hmc_vec: hmc_vec.o
	g++ hmc_vec.o -o hmc_vec
hmc_vec.o: hmc_vec.cpp ../ergo/hmc.h ../ergo/util.h ../ergo/rand.h
	g++ -c hmc_vec.cpp ${CXX_FLAGS} ${INCLUDE_PATHS} -o hmc_vec.o

record: record.o
	g++ record.o -o record
record.o: record.cpp ../ergo/hmc.h ../ergo/record.h ../ergo/util.h ../ergo/rand.h
	g++ -c record.cpp ${CXX_FLAGS} ${INCLUDE_PATHS} -o record.o

clean:
	${RM} *.o mh hmc hmc_vec hmc_rec record