CC = gcc
CFLAGS += -g0 -O3 # -lm -ldl -lc

# Common source files
COMMON_SRCS = algo.c fileutils.c tokenizer.c sampler.c quant.c prune.c

# Profiler sources
PROF_SRCS = mprof.c tprof.c

run: run.o forward.o forwardq.o $(COMMON_SRCS:.c=.o) $(PROF_SRCS:.c=.o)
	$(CC) $(CFLAGS) -o $@ $^

# Rule to compile .c files into .o files
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $^

# Clean rule
clean:
	rm -f run *.o
