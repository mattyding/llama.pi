CC = gcc

CFLAGS += -g0 -O3 -lm

RUN = 0

profiler: profiler.c
	$(CC) $(CFLAGS) $^ -o $@

.PHONY: clean
clean::
	rm -f profiler