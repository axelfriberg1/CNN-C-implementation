#include "./headers/activation.h"
#include "./headers/pool.h"
#include "./headers/utils.h"
#include "headers/backprop.h"
#include "headers/cnn.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define IMG_W 28
#define IMG_H 28
#define CHANNELS 1
#define NUM_FILTERS 6
#define N 70000
#define N_TRAIN 60000 // "max" 60k
#define N_TEST 10000  // "max" 10k
#define BATCH_SIZE 512
#define SEED 42
#define NUM_CLASSES 10
#define LEARNING_RATE 0.01
#define BETA 0.9

static inline double now_sec() {

  // https://stackoverflow.com/questions/3557221/how-do-i-measure-time-in-c
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void shuffle_indices(int *indices, int n) {
  assert(indices);
  for (int i = n - 1; i > 0; i--) {
    int j = rand() % (i + 1);

    int tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
  }
}

/*
    Loads MNIST images based on my Python script:
    float32 normalized values, shape (N, 784)
    Returns:
        double*** images = [N][28][28]
*/
double ***load_mnist_images(const char *filename) {
  FILE *f = fopen(filename, "rb");
  assert(f != NULL);

  // Flat float buffer to read raw file contents
  float *data = malloc(sizeof(float) * N * IMG_H * IMG_W);
  assert(data != NULL);

  const size_t expected = N * IMG_H * IMG_W;
  const size_t read = fread(data, sizeof(float), expected, f);
  fclose(f);
  assert(read == expected);

  // 3D allocation N * IMG_H * IMG_W
  double ***images = malloc(N * sizeof(double **));
  assert(images != NULL);

  for (int i = 0; i < N; i++) {
    images[i] = malloc(IMG_H * sizeof(double *));
    assert(images[i] != NULL);

    for (int y = 0; y < IMG_H; y++) {
      images[i][y] = malloc(IMG_W * sizeof(double));
      assert(images[i][y] != NULL);

      for (int x = 0; x < IMG_W; x++) {
        size_t idx = i * IMG_H * IMG_W + y * IMG_W + x;
        images[i][y][x] = (double)data[idx];
      }
    }
  }

  free(data);
  return images;
}

/*
    Load MNIST labels saved as uint8 array of length N.
*/
unsigned char *load_mnist_labels(const char *filename) {
  FILE *f = fopen(filename, "rb");
  assert(f);

  unsigned char *labels = malloc(N * sizeof(unsigned char));
  assert(labels);

  size_t read = fread(labels, sizeof(unsigned char), N, f);
  fclose(f);

  assert(read == N);

  return labels;
}

void forward(Cnn *cnn, CnnMemory *memory, double **image) {
  conv_forward(cnn->conv1, memory->conv1_buffer, image, IMG_H, IMG_W,
               relu); // out num_filtersx26x26
  pool(cnn->pool1, memory->pool1_buffer, memory->conv1_buffer);
  flatten(memory->pool1_buffer, memory->flat_buffer, NUM_FILTERS,
          cnn->pool1->in_h / 2, cnn->pool1->in_w / 2);
  dense_forw(cnn->dense1, memory->dense1_buffer, memory->flat_buffer, relu);
  dense_forw(cnn->dense2, memory->dense2_buffer, memory->dense1_buffer, relu);
  dense_forw(cnn->dense3, memory->dense3_buffer, memory->dense2_buffer,
             none); // These are logits
  softmax(memory->dense3_buffer, memory->probs_buffer,
          NUM_CLASSES); // probability distribution
}

double backward(Cnn *cnn, CnnMemory *memory, double **image,
                unsigned char label) {

  double loss = softmax_cross_entropy_back(memory->probs_buffer, label,
                                           memory->grad_logits, NUM_CLASSES);

  dense_back_no_activation(cnn->dense3, memory->dense2_buffer,
                           memory->grad_logits, memory->grad_dense2);
  dense_back_relu(cnn->dense2, memory->dense2_buffer, memory->dense1_buffer,
                  memory->grad_dense2, memory->grad_dense1);
  dense_back_relu(cnn->dense1, memory->dense1_buffer, memory->flat_buffer,
                  memory->grad_dense1, memory->grad_flat);

  unflatten(memory->grad_flat, memory->grad_pool,
            NUM_FILTERS * cnn->pool1->in_h / 2 * cnn->pool1->in_w / 2,
            cnn->pool1->num_fmaps, cnn->pool1->in_h / 2, cnn->pool1->in_w / 2);

  pool_back(cnn->pool1, memory->conv1_buffer, memory->pool1_buffer,
            memory->grad_pool, memory->grad_conv);
  conv_back(cnn->conv1, image, memory->conv1_buffer, memory->grad_conv,
            memory->grad_input);

  return loss;
}

void train(Cnn *cnn, CnnMemory *memory, double ***images, unsigned char *labels,
           int epochs, int *train_idx) {

  double sum_loss = 0;

  for (int epoch = 0; epoch < epochs; epoch++) {
    double t10 = now_sec();
    int batch_count = 0;

    shuffle_indices(train_idx, N_TRAIN);

    for (int k = 0; k < N_TRAIN; k++) {
      int i = train_idx[k];
      unsigned char label = labels[i];
      double **image = images[i];
      batch_count++;

      forward(cnn, memory, image);
      sum_loss += backward(cnn, memory, image, label);

      cnn_zero_backward_buffers(cnn, memory);

      // batch boundary
      if (batch_count == BATCH_SIZE || k == N_TRAIN - 1) {

        // scale here because gradients are just summed, not averaged.
        double scale = 1 / (double)batch_count;
        batch_count = 0;

        update_conv(cnn->conv1, LEARNING_RATE * scale, BETA);
        update_dense(cnn->dense1, LEARNING_RATE * scale, BETA);
        update_dense(cnn->dense2, LEARNING_RATE * scale, BETA);
        update_dense(cnn->dense3, LEARNING_RATE * scale, BETA);

        zero_conv_gradients(cnn->conv1);
        zero_dense_gradients(cnn->dense1);
        zero_dense_gradients(cnn->dense2);
        zero_dense_gradients(cnn->dense3);
      }
    }

    double t11 = now_sec();

    printf("Epoch %d avgLoss = %f\n", epoch, sum_loss / N_TRAIN);

    if (sum_loss / N_TRAIN < 0.2) {
      printf("training stopped");
      cnn_zero_forward_buffers(cnn, memory);
      cnn_zero_backward_buffers(cnn, memory);
      break;
    }

    sum_loss = 0;
  }
}

void test(Cnn *cnn, CnnMemory *memory, double ***images, unsigned char *labels,
          int *test_idx) {

  shuffle_indices(test_idx, N_TEST);
  int correct = 0;

  for (int k = 0; k < N_TEST; k++) {
    int i = test_idx[k];

    double **image = images[i];
    unsigned char label = labels[i];

    forward(cnn, memory, image);
    for (int i = 0; i < 10; i++) {
      printf("%.3f ", memory->probs_buffer[i]);
    }
    printf("\n");
    // argmax over probabilities
    int pred = 0;
    double max_prob = memory->probs_buffer[0];
    for (int j = 1; j < NUM_CLASSES; j++) {
      if (memory->probs_buffer[j] > max_prob) {
        max_prob = memory->probs_buffer[j];
        pred = j;
      }
    }

    if (pred == label) {
      correct++;
    }

    double loss = cross_entropy(memory->probs_buffer, label, NUM_CLASSES);
    cnn_zero_forward_buffers(cnn, memory);

    printf("Label: %d Loss: %f\n", label, loss);
  }
  double accuracy = (double)correct / N_TEST;
  printf("Test accuracy: %.4f\n", accuracy);
}

void Evaluate(Cnn *cnn, CnnMemory *memory, double ***images,
              unsigned char *labels, int epochs) {

  // setting up different views for training and test data
  int *train_idx = malloc(sizeof(int) * N_TRAIN);
  int *test_idx = malloc(sizeof(int) * N_TEST);
  for (int i = 0; i < N_TRAIN; i++)
    train_idx[i] = i;
  for (int i = 0; i < N_TEST; i++)
    test_idx[i] = N_TRAIN + i;

  train(cnn, memory, images, labels, epochs, train_idx);
  test(cnn, memory, images, labels, test_idx);

  free(train_idx);
  free(test_idx);
}

int main() {
  // This blog has been of help with conceptual understanding of each step:
  // https://victorzhou.com/blog/intro-to-cnns-part-2/ Statquest youtube channel
  // has also been of great help with understanding backpropagation and its
  // implementation

  // TODO: checkout im2col + gemm

  srand(SEED);

  // load datasets, preprocessed in python
  // these datasets will not be split or modified
  // instead different views are ensured through indexing
  const char images_path[85] = "./cnn/resources/mnist_images.bin";
  const char labels_path[85] = "./cnn/resources/mnist_labels.bin";
  double ***images = load_mnist_images(images_path);
  unsigned char *labels = load_mnist_labels(labels_path);

  ConvLayer *conv_layer = init_convlayer(NUM_FILTERS, 3);
  PoolingLayer *pooling_layer = init_pooling(26, 26, 2, NUM_FILTERS);
  DenseLayer *dense_layer1 = init_dense(NUM_FILTERS * 13 * 13, 32);
  DenseLayer *dense_layer2 = init_dense(32, 16);
  DenseLayer *output_layer = init_dense(16, 10);

  Cnn *cnn = init_cnn(conv_layer, pooling_layer, dense_layer1, dense_layer2,
                      output_layer, IMG_H, IMG_W);

  CnnMemory *memory = init_memory(cnn);

  Evaluate(cnn, memory, images, labels, 200);

  /*free_convlayer(conv_layer);
  free_pool(pooling_layer);
  free_dense(dense_layer1);
  free_dense(dense_layer2);
  free_dense(output_layer);
  free_memory(memory, cnn);*/

  /* Free dataset */
  free_3d_array(images, N, IMG_H);
  free(labels);

  return 0;
}
