BOOTLOADER = my-install

COMMON_SRC += util.c
COMMON_SRC += algo.c
COMMON_SRC += quant.c

PROGS += runq-seg.c

# define this if you need to give the device for your pi
TTYUSB = 

# set RUN = 1 if you want the code to automatically run after building.
RUN = 1

LIBS += $(PI_LLAMA_PATH)/lib/libm.a
LIBS += $(PI_LLAMA_PATH)/lib/libc.a
LIBS += $(PI_LLAMA_PATH)/lib/libgcc.a

# # semihosting (https://stackoverflow.com/questions/19419782/exit-c-text0x18-undefined-reference-to-exit-when-using-arm-none-eabi-gcc)
CFLAGS += --specs=nosys.specs
# -fno-math-errno doesn't work
# -ffreestanding, -nostdlib already included

# # link to math functions
CFLAGS += -lm -lc -lgcc

# external-code dir
CFLAGS  += -Iexternal-code
COMMON_SRC += external-code/unicode-utf8.c external-code/emmc.c external-code/mbox.c 

include $(PI_LLAMA_PATH)/libpi/mk/Makefile.robust

clean::
	make -C clean