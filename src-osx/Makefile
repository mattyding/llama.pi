CC = gcc

CFLAGS += -g0 -O3 -lm -ldl -lc

RUN = 0

all: profiler profilerq

profilerq: profiler.c prof-runq.c
	$(CC) -c -o profilerq.o profiler.c
	$(CC) $(CFLAGS) prof-runq.c profilerq.o -o profilerq

profiler: profiler.c prof-run.c
	$(CC) -c -o profiler.o profiler.c
	$(CC) $(CFLAGS) prof-run.c profiler.o -o profiler

.PHONY: clean
clean:
	rm -f profiler profiler.o
	rm -f profilerq profiler.o