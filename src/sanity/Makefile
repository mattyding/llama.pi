BOOTLOADER = my-install

O = $(PI_LLAMA_PATH)/libpi

PROGS += hello.c

# define this if you need to give the device for your pi
TTYUSB = 

# set RUN = 1 if you want the code to automatically run after building.
RUN = 1

LIBS += $(PI_LLAMA_PATH)/lib/libgcc.a

include $(PI_LLAMA_PATH)/libpi/mk/Makefile.robust

clean::
	make -C clean
