#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#define WIDTH 72
#define HEIGHT 34
#define MAX_SNAKE (WIDTH * HEIGHT)

struct point { int x; int y; };
static struct termios original_terminal;
static volatile sig_atomic_t stop_requested;

static void restore_terminal(void) {
  tcsetattr(STDIN_FILENO, TCSAFLUSH, &original_terminal);
  printf("\033[0m\033[?25h\033[%d;1H\n", HEIGHT + 3);
  fflush(stdout);
}

static void request_stop(int signal_number) {
  (void)signal_number;
  stop_requested = 1;
}

static int read_key(void) {
  fd_set input;
  FD_ZERO(&input);
  FD_SET(STDIN_FILENO, &input);
  struct timeval timeout = {0, 80000};
  if (select(STDIN_FILENO + 1, &input, NULL, NULL, &timeout) <= 0)
    return 0;
  unsigned char key;
  if (read(STDIN_FILENO, &key, 1) != 1)
    return 0;
  if (key != 27)
    return key;
  unsigned char sequence[2];
  if (read(STDIN_FILENO, sequence, 2) != 2 || sequence[0] != '[')
    return 0;
  return 256 + sequence[1];
}

static unsigned random_state = 0x2050c0de;
static unsigned random_number(void) {
  random_state ^= random_state << 13;
  random_state ^= random_state >> 17;
  random_state ^= random_state << 5;
  return random_state;
}

static int occupies(const struct point *snake, int length, int x, int y) {
  for (int index = 0; index < length; index++)
    if (snake[index].x == x && snake[index].y == y)
      return 1;
  return 0;
}

int main(void) {
  if (tcgetattr(STDIN_FILENO, &original_terminal) != 0)
    return 1;
  struct termios raw = original_terminal;
  raw.c_lflag &= (tcflag_t)~(ECHO | ICANON);
  raw.c_cc[VMIN] = 1;
  raw.c_cc[VTIME] = 0;
  tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
  atexit(restore_terminal);
  signal(SIGINT, request_stop);
  signal(SIGTERM, request_stop);

  struct point snake[MAX_SNAKE];
  int length = 6;
  for (int index = 0; index < length; index++)
    snake[index] = (struct point){WIDTH / 2 - index, HEIGHT / 2};
  int dx = 1, dy = 0;
  struct point food = {WIDTH / 4, HEIGHT / 3};
  int score = 0;

  printf("\033[2J\033[H\033[?25l\033[1;36mFPGA Snake\033[0m  arrows/WASD move, q quits\n");
  for (int y = 0; y <= HEIGHT; y++) {
    printf("\033[%d;1H\033[34m#", y + 2);
    printf("\033[%d;%dH#\033[0m", y + 2, WIDTH + 2);
  }
  for (int x = 0; x <= WIDTH + 1; x++) {
    printf("\033[2;%dH\033[34m#", x + 1);
    printf("\033[%d;%dH#\033[0m", HEIGHT + 2, x + 1);
  }

  while (!stop_requested) {
    int key = read_key();
    if (key == 'q' || key == 'Q')
      break;
    if ((key == 'w' || key == 256 + 'A') && dy != 1) { dx = 0; dy = -1; }
    if ((key == 's' || key == 256 + 'B') && dy != -1) { dx = 0; dy = 1; }
    if ((key == 'd' || key == 256 + 'C') && dx != -1) { dx = 1; dy = 0; }
    if ((key == 'a' || key == 256 + 'D') && dx != 1) { dx = -1; dy = 0; }

    struct point next = {snake[0].x + dx, snake[0].y + dy};
    if (next.x < 0 || next.x >= WIDTH || next.y < 0 || next.y >= HEIGHT || occupies(snake, length - 1, next.x, next.y))
      break;

    struct point old_tail = snake[length - 1];
    memmove(&snake[1], &snake[0], (size_t)length * sizeof(*snake));
    snake[0] = next;
    if (next.x == food.x && next.y == food.y) {
      if (length < MAX_SNAKE)
        snake[length++] = old_tail;
      score++;
      do {
        food.x = (int)(random_number() % WIDTH);
        food.y = (int)(random_number() % HEIGHT);
      } while (occupies(snake, length, food.x, food.y));
    } else {
      printf("\033[%d;%dH ", old_tail.y + 3, old_tail.x + 2);
    }
    printf("\033[%d;%dH\033[1;32m@\033[0m", snake[0].y + 3, snake[0].x + 2);
    if (length > 1)
      printf("\033[%d;%dH\033[32mo\033[0m", snake[1].y + 3, snake[1].x + 2);
    printf("\033[%d;%dH\033[1;31m*\033[0m", food.y + 3, food.x + 2);
    printf("\033[1;%dHscore: %d", WIDTH - 10, score);
    fflush(stdout);
  }

  printf("\033[%d;1H\033[1;33mGame over — score %d.\033[0m", HEIGHT + 3, score);
  return 0;
}
