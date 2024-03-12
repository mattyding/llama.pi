#include "util.h"

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

char *read_file(fat32_fs_t *fs, pi_dirent_t *dir, char *name) {
  pi_file_t *f = fat32_read(fs, dir, name);
  assert(f != NULL);
  return f->data;
}

void init_fs(fat32_fs_t *fs, pi_dirent_t *root) {
  mbr_t *mbr = mbr_read();
  mbr_partition_ent_t partition;
  memcpy(&partition, mbr->part_tab1, sizeof(mbr_partition_ent_t));
  assert(mbr_part_is_fat32(partition.part_type));
  *fs = fat32_mk(&partition); // load fat
  *root = fat32_get_root(fs);
}

pi_dirent_t *init_fs_cd_model(fat32_fs_t *fs, pi_dirent_t *root) {
  init_fs(fs, root);
  return cd(fat32_readdir(fs, root), *fs, MODEL_DIR);
}