BOOTLOADER = my-install

O = $(PI_LLAMA_PATH)/libpi

# PROGS += $(wildcard *.c)
PROGS += run.c

# define this if you need to give the device for your pi
TTYUSB = 

# set RUN = 1 if you want the code to automatically run after building.
RUN = 1

LIBS += $(PI_LLAMA_PATH)/lib/libm.a
# LIBS += $(PI_LLAMA_PATH)/lib/libc.a
LIBS += $(PI_LLAMA_PATH)/lib/libgcc.a

# semihosting (https://stackoverflow.com/questions/19419782/exit-c-text0x18-undefined-reference-to-exit-when-using-arm-none-eabi-gcc)
CFLAGS += --specs=nosys.specs
CFLAGS += -ffreestanding
# LFLAGS += -ffreestanding	# don't think these are necessary
LFLAGS += -nostdlib 
# LFLAGS += -shared			# ffreestanding requires shared

# link to math functions
CFLAGS += -lm -lgcc

include $(PI_LLAMA_PATH)/libpi/mk/Makefile.robust

clean::
	make -C clean
