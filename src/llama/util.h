#ifndef __LLAMA_UTIL_H__
#define __LLAMA_UTIL_H__
#include "rpi.h"
#include "pi-sd.h"
#include "fat32.h"
#include "transformer.h"

#define MODEL_DIR "7B_Q80"  // weight dir on SD card

void rpi_init(void);

// --- fat32 fs helpers ---

void ls(pi_directory_t files);
pi_dirent_t *cd(pi_directory_t cwd, fat32_fs_t fs, char *name);
char *read_file(fat32_fs_t *fs, pi_dirent_t *dir, char *name);

void init_fs(fat32_fs_t *fs, pi_dirent_t *root);
pi_dirent_t *init_fs_cd_model(fat32_fs_t *fs, pi_dirent_t *root);

#endif