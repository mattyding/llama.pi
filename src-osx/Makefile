CC = gcc
CFLAGS += -g0 -O3 # -lm -ldl -lc

# Common source files
COMMON_SRCS = algo.c fileutils.c tokenizer.c sampler.c

# Profiler sources
PROF_SRCS = mprof.c tprof.c

run: run.o forward.o $(COMMON_SRCS:.c=.o) $(PROF_SRCS:.c=.o)
	$(CC) $(CFLAGS) -o $@ $^

runq: runq.o forwardq.o quant.o $(COMMON_SRCS:.c=.o) $(PROF_SRCS:.c=.o)
	$(CC) $(CFLAGS) -o $@ $^


run.o forward.o runq.o forwardq.o quant.o $(COMMON_SRCS:.c=.o): %.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(PROF_SRCS:.c=.o): %.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Clean rule
clean:
	rm -f run runq *.o
