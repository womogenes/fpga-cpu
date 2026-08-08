/* F-only userspace uses the lp64 soft-float ABI; give TinyCC (which
   assumes lp64d) the same stubs so native builds work. */
#include <gnu/stubs-lp64.h>
