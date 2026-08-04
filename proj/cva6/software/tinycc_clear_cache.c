void __clear_cache(void *begin, void *end) {
  (void)begin;
  (void)end;
  __asm__ volatile("fence.i" ::: "memory");
}
