#include "algo.h"
#include "transformer.h"
#include "forwardq.c"
#include "util.h"

#include "rpi.h"
#include "cycle-count.h"

int data_type_size = sizeof(float);

void benchmark_matmulf(Config *p, int idx) {
    // hard coding these in because of poss overflow errors
    int dim = 4096;
    int kv_dim = 4096; // dim * n_kv_heads / n_heads
    int head_size = 128; // dim / n_heads
    int hidden_dim = 11008;
    int n_layers = 32;

    float *mat1;
    float *mat2;
    int dim1;
    int dim2;
    float one_iter_scale = head_size; // if OOM, scale down mat2 and multiply final time by one_iter_scale

    switch (idx) {
        case 0:
            // s->q, s->xb, w->wq
            // dims: dim, dim
            printk("s->q matmulf: this happens %d (n_layers) times.\n", n_layers); 
            mat1 = kmalloc(dim * data_type_size);
            mat2 = kmalloc(dim * data_type_size);
            dim1 = dim;
            dim2 = dim;
            printk("one iter scale: 128\n");
            break;
        case 1:
            // s->k, s->xb, w->wk
            // dims: dim, kv_dim
            printk("s->k matmulf: this happens %d (n_layers) times.\n", n_layers);
            mat1 = kmalloc(dim * data_type_size);
            mat2 = kmalloc(kv_dim * data_type_size);
            dim1 = dim;
            dim2 = kv_dim;
            printk("one iter scale: 128\n");
            break;
        case 2:
            // s->v, s->xb, w->wv
            // dims: dim, kv_dim
            printk("s->v matmulf: this happens %d (n_layers) times.\n", n_layers);
            mat1 = kmalloc(dim * data_type_size);
            mat2 = kmalloc(kv_dim * data_type_size);
            dim1 = dim;
            dim2 = kv_dim;
            printk("one iter scale: 128\n");
            break;
        case 3:
            // s->xb2, s->xb, w->wo
            // dims: dim, dim
            printk("s->xb2 matmulf: this happens %d (n_layers) times.\n", n_layers);
            mat1 = kmalloc(dim * data_type_size);
            mat2 = kmalloc(dim * data_type_size);
            dim1 = dim;
            dim2 = dim;
            printk("one iter scale: 128\n");
            break;
        case 4:
            // s->hb, s->xb, w->w1
            // dims: dim, hidden_dim
            printk("s->hb matmulf: this happens %d (n_layers) times.\n", n_layers);
            mat1 = kmalloc(dim * data_type_size);
            mat2 = kmalloc(hidden_dim * data_type_size);
            dim1 = dim;
            dim2 = hidden_dim;
            printk("one iter scale: 4096\n");
            break;
        case 5:
            // s->hb2, s->xb, w->w3
            // dims: dim, hidden_dim
            printk("s->hb2 matmulf: this happens %d (n_layers) times.\n", n_layers);
            mat1 = kmalloc(dim * data_type_size);
            mat2 = kmalloc(hidden_dim * data_type_size);
            dim1 = dim;
            dim2 = hidden_dim;
            printk("one iter scale: 4096\n");
            break;
        case 6:
            // s->xb, s->hb, w->w2
            // dims: hidden_dim, dim
            printk("s->xb matmulf: this happens %d (n_layers) times.\n", n_layers);
            mat1 = kmalloc(hidden_dim * data_type_size);
            mat2 = kmalloc(hidden_dim * dim * data_type_size);
            dim1 = hidden_dim;
            dim2 = dim;
            printk("one iter scale: 11008\n");
            break;
        case 7:
            // s->logits, x, token_embedding_table
            // dims: dim, p->vocab_size
            printk("s->logits matmulf: this happens once per pass.\n");
            mat1 = kmalloc(dim * data_type_size);
            mat2 = kmalloc(p->vocab_size * data_type_size);
            dim1 = dim;
            dim2 = p->vocab_size;
            printk("one iter scale: 4096\n");
            break;
        // else
        default:
            panic("benchmark_matmulf: invalid idx");
            break;
    }
    float *out = mat1;  // works because out[i] is assigned after done using mat1[i]
    
    
    // time the matmulf
    cycle_cnt_init();
    uint32_t start_time = timer_get_usec();
    unsigned start_cyc = cycle_cnt_read();
    matmulf(out, mat1, mat2, dim1, dim2);
    unsigned end_cyc = cycle_cnt_read();
    uint32_t end_time = timer_get_usec();

    // I found a slight discrepency in timing. might be due to overflow or imprecision?
    // consulting the docs, from BCM2835 ARM Peripherals, page 196:
    // timer_clock = apb_clock/(pre_divider+1)
    // for instance, clock reports 22sec but my phone only measures 17sec

    printk("start_time: %u\n", start_time);
    printk("end_time: %u\n", end_time);

    // this seems to be relatively fine.
    unsigned cycles = (end_cyc - start_cyc);
    printk("one matmulf took %u cycles\n", cycles);

    printk("don't forget to multiply by one_iter_scale\n");
}

void benchmark_load_layerq(Config *p) {
    printk("benchmarking one quantized layer weight load\n");
    rpi_init();
    cycle_cnt_init();

    // init filesys and cd to model dir
    pi_dirent_t root;
    fat32_fs_t fs;
    pi_dirent_t w_dir = *init_fs_cd_model(&fs, &root);

    qLayerWeights w;
    // we OOM when mallocing layers so we simulate reading in the file size
    // malloc_layer_weights(p, &w); // set w
    qRunState s;
    // malloc_run_state(p, &s); // set s

    uint32_t start_time = timer_get_usec();
    unsigned start_cyc = cycle_cnt_read();

    pi_file_t *f = fat32_read(&fs, &w_dir, "layer0.bin");
    int f_size = f->n_data;
    char buf[f_size / 8];
    for (int i = 0; i < f_size / 8; i++) {
        buf[i] = f->data[i];
    }
    printk("%s", buf[0]); // so compiler doesn't complain

    unsigned end_cyc = cycle_cnt_read();
    uint32_t end_time = timer_get_usec();

    printk("start_time: %u\n", start_time);
    printk("end_time: %u\n", end_time);

    // this seems to be relatively fine.
    unsigned cycles = (end_cyc - start_cyc);
    printk("layer weight load took %u cycles.\n", cycles);
}

void notmain(void) {
    Config config;
    // loaded model: dim=4096, hidden_dim=11008, n_layers=32, n_heads=32, n_kv_heads=32, vocab_size=32000, seq_len=2048
    config.dim = 4096;
    config.hidden_dim = 11008;
    config.n_layers = 32;
    config.n_heads = 32;
    config.n_kv_heads = 32;
    config.vocab_size = 32000;
    config.seq_len = 2048;

    // benchmark_matmulf(&config, 0);
    // benchmark_matmulf(&config, 1);
    // benchmark_matmulf(&config, 2);
    // benchmark_matmulf(&config, 3);
    // benchmark_matmulf(&config, 4);
    // benchmark_matmulf(&config, 5);
    // benchmark_matmulf(&config, 6);

    // benchmark_load_layerq(&config);
}