# setup useful variables that can be used by make.

# this must be defined in your shell's startup file.
ifndef PI_LLAMA_PATH
$(error PI_LLAMA_PATH is not set: this should contain the absolute path to where this directory is.  Define it in your shell's initialiation.  For example, .tcshrc for tcsh or .bashrc for bash)
endif

# OPT_LEVEL = -O3
ARM = arm-none-eabi
CC = $(ARM)-gcc
LD  = $(ARM)-ld
AS  = $(ARM)-as
AR = $(ARM)-ar
OD  = $(ARM)-objdump
OCP = $(ARM)-objcopy
PI_LLAMA_LIBPI_PATH = $(PI_LLAMA_PATH)/libpi
LPP = $(PI_LLAMA_LIBPI_PATH)
LPI ?= $(LPP)/libpi.a
LGCC ?= $(PI_LLAMA_PATH)/lib/libgcc.a

# let the client override these.
START ?= $(LPP)/staff-start.o
DEFAULT_START := $(LPP)/staff-start.o
MEMMAP ?= $(LPP)/memmap

# include path: user can override
INC += -I. -I$(LPP)/include -I$(LPP)/ -I$(LPP)/src  -I$(LPP)/libc -I$(LPP)/staff-private
# optimization level: client can override
OPT_LEVEL ?= -Og

# various warnings.  any warning = error, so you must fix it.
CFLAGS += -D__RPI__ $(OPT_LEVEL) -Wall -nostdlib -nostartfiles -ffreestanding -mcpu=arm1176jzf-s -mtune=arm1176jzf-s  -std=gnu99 $(INC) -ggdb -Wno-pointer-sign  -Werror  -Wno-unused-function -Wno-unused-variable 

ifdef CFLAGS_EXTRA
CFLAGS += CFLAGS_EXTRA
endif

# -ffunction-sections -fdata-sections -Wl,--gc-sections

# workaround for gcc struct assignment bug 
CFLAGS += -mno-unaligned-access

# disable the thread-local pointer coprocessor register (for Thumb support)
CFLAGS += -mtp=soft

# for backtrace
# CFLAGS += -fno-omit-frame-pointer -mpoke-function-name -DBACKTRACE


# for assembly file compilation.
# ASFLAGS = --warn --fatal-warnings  -mcpu=arm1176jzf-s -march=armv6zk $(INC)

# for .S compilation so we can use the preprocessor.
CPP_ASFLAGS =  -nostdlib -nostartfiles -ffreestanding   -Wa,--warn -Wa,--fatal-warnings -Wa,-mcpu=arm1176jzf-s -Wa,-march=armv6zk   $(INC)
