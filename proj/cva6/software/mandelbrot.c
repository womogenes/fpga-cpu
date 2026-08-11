#include "video_hw.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#define MAX_ITERATIONS 48u
#define RENDER_THREADS 4u
#define ZOOM_FACTOR 0.992f
/* Below this span, adjacent 160-pixel samples approach the float ULP around
   the chosen center and the picture degenerates into repeated columns. */
#define MIN_SPAN 0.000020f

static const float center_real = -0.7436439f;
static const float center_imag = 0.1318259f;
static volatile sig_atomic_t stop_requested;

struct render_pool {
  pthread_barrier_t begin;
  pthread_barrier_t done;
  pthread_t threads[RENDER_THREADS - 1u];
  uint8_t *frame;
  float span;
  int stop;
};

struct worker_arg {
  struct render_pool *pool;
  unsigned index;
};

static struct worker_arg worker_args[RENDER_THREADS - 1u];

static void request_stop(int signal_number) {
  (void)signal_number;
  stop_requested = 1;
}

static void poll_stop_key(void) {
  unsigned char key;
  ssize_t count;
  while ((count = read(STDIN_FILENO, &key, 1)) == 1)
    if (key == 'q' || key == 'Q' || key == 3u)
      stop_requested = 1;
  if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
    stop_requested = 1;
}

static uint8_t palette(unsigned iteration) {
  if (iteration == MAX_ITERATIONS)
    return 0;
  /* A nonlinear phase avoids the nearly identical adjacent colors produced
     by simply truncating a linear 8-bit ramp to RGB232. */
  unsigned phase = iteration * iteration + iteration * 13u;
  unsigned red = phase * 5u;
  unsigned green = phase * 3u + 64u;
  unsigned blue = phase * 7u + 128u;
  return video_rgb232(red & 255u, green & 255u, blue & 255u);
}

static uint8_t sample(float cr, float ci) {
  /* Main cardioid and period-two bulb tests remove most expensive interior
     points without changing the image. */
  float shifted = cr - 0.25f;
  float ci2 = ci * ci;
  float q = shifted * shifted + ci2;
  if (q * (q + shifted) <= 0.25f * ci2)
    return 0;
  float bulb = cr + 1.0f;
  if (bulb * bulb + ci2 <= 0.0625f)
    return 0;

  float zr = 0.0f;
  float zi = 0.0f;
  unsigned iteration = 0;
  while (iteration < MAX_ITERATIONS) {
    float zr2 = zr * zr;
    float zi2 = zi * zi;
    if (zr2 + zi2 > 4.0f)
      break;
    zi = 2.0f * zr * zi + ci;
    zr = zr2 - zi2 + cr;
    ++iteration;
  }
  return palette(iteration);
}

static void render_rows(struct render_pool *pool, unsigned worker) {
  float span = pool->span;
  float vertical_span = span * (float)VIDEO_HEIGHT / (float)VIDEO_WIDTH;
  float real_step = (2.0f * span) / (float)VIDEO_WIDTH;
  float imag_step = (2.0f * vertical_span) / (float)VIDEO_HEIGHT;
  float first_real = center_real - span + 0.5f * real_step;
  float first_imag = center_imag - vertical_span + 0.5f * imag_step;

  for (unsigned y = worker; y < VIDEO_HEIGHT; y += RENDER_THREADS) {
    float ci = first_imag + (float)y * imag_step;
    uint8_t *row = pool->frame + y * VIDEO_WIDTH;
    for (unsigned x = 0; x < VIDEO_WIDTH; ++x)
      row[x] = sample(first_real + (float)x * real_step, ci);
  }
}

static void *render_worker(void *opaque) {
  struct worker_arg *arg = opaque;
  for (;;) {
    (void)pthread_barrier_wait(&arg->pool->begin);
    if (arg->pool->stop)
      break;
    render_rows(arg->pool, arg->index);
    (void)pthread_barrier_wait(&arg->pool->done);
  }
  return NULL;
}

static int render_pool_open(struct render_pool *pool, uint8_t *frame) {
  pool->frame = frame;
  if (pthread_barrier_init(&pool->begin, NULL, RENDER_THREADS) != 0 ||
      pthread_barrier_init(&pool->done, NULL, RENDER_THREADS) != 0)
    return 1;
  for (unsigned i = 1; i < RENDER_THREADS; ++i) {
    worker_args[i - 1u].pool = pool;
    worker_args[i - 1u].index = i;
    if (pthread_create(&pool->threads[i - 1u], NULL, render_worker,
                       &worker_args[i - 1u]) != 0) {
      pool->stop = 1;
      /* A partial pool cannot satisfy the barrier, so terminate cleanly
         instead of entering it. The process exit reclaims those threads. */
      return 1;
    }
  }
  return 0;
}

static void render_frame(struct render_pool *pool, float span) {
  pool->span = span;
  (void)pthread_barrier_wait(&pool->begin);
  render_rows(pool, 0u);
  (void)pthread_barrier_wait(&pool->done);
}

static void render_pool_close(struct render_pool *pool) {
  pool->stop = 1;
  (void)pthread_barrier_wait(&pool->begin);
  for (unsigned i = 1; i < RENDER_THREADS; ++i)
    (void)pthread_join(pool->threads[i - 1u], NULL);
  (void)pthread_barrier_destroy(&pool->done);
  (void)pthread_barrier_destroy(&pool->begin);
}

static double elapsed_seconds(const struct timespec *start,
                              const struct timespec *end) {
  return (double)(end->tv_sec - start->tv_sec) +
         (double)(end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

int main(void) {
  struct video_hw video = {0};
  if (video_hw_open(&video) != 0)
    return 1;

  uint8_t *frame = malloc(VIDEO_PIXELS);
  struct render_pool pool = {0};
  if (frame == NULL || render_pool_open(&pool, frame) != 0) {
    free(frame);
    video_hw_close(&video);
    return 1;
  }

  signal(SIGINT, request_stop);
  signal(SIGTERM, request_stop);
  struct termios saved_terminal;
  int terminal_valid = tcgetattr(STDIN_FILENO, &saved_terminal) == 0;
  int saved_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
  if (terminal_valid) {
    struct termios raw = saved_terminal;
    cfmakeraw(&raw);
    (void)tcsetattr(STDIN_FILENO, TCSANOW, &raw);
  }
  if (saved_flags >= 0)
    (void)fcntl(STDIN_FILENO, F_SETFL, saved_flags | O_NONBLOCK);
  struct timespec started;
  clock_gettime(CLOCK_MONOTONIC, &started);
  unsigned frames = 0;
  float span = 1.6f;
  int status = 0;

  render_frame(&pool, span);
  while (!stop_requested) {
    poll_stop_key();
    if (stop_requested)
      break;
    if (video_hw_submit(&video, frame) != 0) {
      status = 1;
      break;
    }
    ++frames;
    span *= ZOOM_FACTOR;
    if (span < MIN_SPAN)
      span = 1.6f;
    render_frame(&pool, span);
  }

  struct timespec finished;
  clock_gettime(CLOCK_MONOTONIC, &finished);
  render_pool_close(&pool);
  video_hw_terminal(&video);
  video_hw_close(&video);
  if (saved_flags >= 0)
    (void)fcntl(STDIN_FILENO, F_SETFL, saved_flags);
  if (terminal_valid)
    (void)tcsetattr(STDIN_FILENO, TCSANOW, &saved_terminal);
  free(frame);
  double seconds = elapsed_seconds(&started, &finished);
  if (seconds > 0.0)
    printf("Rendered %u coherent frames in %.2f seconds (%.2f fps).\n",
           frames, seconds, frames / seconds);
  return status;
}
