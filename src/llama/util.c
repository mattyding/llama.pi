#include "rpi.h"
#include "pi-sd.h"
#include "fat32.h"

void rpi_init(void) {
    uart_init();
    kmalloc_init();
    pi_sd_init();
}

void ls(pi_directory_t files) {
  for (int i = 0; i < files.ndirents; i++) {
    if (files.dirents[i].is_dir_p) {
      printk("\tD: %s (cluster %d)\n", files.dirents[i].name, files.dirents[i].cluster_id);
    } else {
      printk("\tF: %s (cluster %d; %d bytes)\n", files.dirents[i].name, files.dirents[i].cluster_id, files.dirents[i].nbytes);
    }
  }
}

#define min(a, b) ((a) < (b) ? (a) : (b))

pi_dirent_t *cd(pi_directory_t cwd, fat32_fs_t fs, char *name) {
  for (int i = 0; i < cwd.ndirents; i++) {
    if (strncmp(cwd.dirents[i].name, name, min(strlen(name), 16)) == 0) {
      return &cwd.dirents[i];
    }
  }
  panic("Directory %s not found\n", name);
}