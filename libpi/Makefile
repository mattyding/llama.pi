# simple definitions for making libpi.a
#  - you should only ever have to modify SRC or STAFF_OBJS

# c files to compile in /libpi/src
SRC += src/fat32-helpers.c
SRC += src/fat32-lfn-helpers.c
SRC += src/mbr-helpers.c
SRC += src/pi-sd.c

# object files to link in /libpi/staff-objs
STAFF_OBJS  +=  ./staff-objs/gpio.o
STAFF_OBJS  +=  ./staff-objs/uart.o
STAFF_OBJS  +=  ./staff-objs/gpio-pud.o
STAFF_OBJS  +=  ./staff-objs/gpio-int.o
STAFF_OBJS  +=  ./staff-objs/fat32.o
STAFF_OBJS  +=  ./staff-objs/mbr.o


# you should not have to modify this variable.  
SRC :=  $(SRC)                              \
        $(wildcard ./libc/*.[Sc])           \
        $(wildcard ./staff-src/*.[Sc])      \
        staff-start.S

DEPS += ./Makefile ./manifest.mk
COMMON_SRC = $(SRC)

# hack to minimize git conflicts: we do various customizations
# in there; but probably would be clearer to inline it.
include ./manifest.mk
