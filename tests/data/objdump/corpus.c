/* Corpus for the objdump byte-column property harness (issue #65 spike P4).
   One file, per-arch blocks. Built with clang --target=... -O2 -c. */
#include <stdint.h>

int sw(int x, int y) {          /* dense switch: jump table (tbb/tbh in Thumb-2) */
    switch (x) {
    case 0: return y + 1;   case 1: return y * 3;  case 2: return y - 7;
    case 3: return y ^ 5;   case 4: return y << 2; case 5: return y >> 1;
    case 6: return y | 9;   case 7: return y & 3;  case 8: return -y;
    case 9: return y / 3;   default: return 0;
    }
}

uint64_t big(void) { return 0x1122334455667788ull; }   /* movabs / literal or movz chain */

double fadd3(double a, double b, double c) { return a + b + c; }

void bytes8(volatile char *p) { p[0] += 1; p[0x11223344] += 2; }

void loop(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] = a[i] * 2.0f + b[i];
}

#if defined(__x86_64__)
#include <immintrin.h>
__attribute__((target("avx512f")))
void evex(float *p) {   /* EVEX with 32-bit displacement */
    __m512 v = _mm512_loadu_ps(p + 0x12345);
    v = _mm512_add_ps(v, _mm512_loadu_ps(p + 0x40000));
    _mm512_storeu_ps(p + 0x23456, v);
    __m512i a = _mm512_loadu_si512(p), b = _mm512_loadu_si512(p + 16);
    _mm512_storeu_si512(p + 0x30000, _mm512_ternarylogic_epi32(a, b, a, 0x96));
}
void longenc(void) {
    __asm__ volatile(
        "movabsq $0x1122334455667788, %%rax\n"
        "lock addl $0x11223344, %%fs:0x11223344(%%rax)\n"
        "movl $0x11223344, 0x11223344(%%rax,%%rbx,4)\n"
        "fadd %%st(1), %%st\n"
        "addb $1, (%%rax)\n"
        "data16 cs nopw 0x0(%%rax,%%rax,1)\n"
        "rep stosb\n"
        "vpternlogd $0x96, 0x40(%%rax,%%rbx,4), %%zmm1, %%zmm2\n"
        ".byte 0x0f, 0x0b\n"   /* raw bytes in text (ud2) */
        ::: "rax", "memory");
}
long double x87(long double a, long double b) { return a + b; }
#elif defined(__aarch64__)
void dat(void) { __asm__ volatile("nop\n.word 0x12345678\n.word 0xdeadbeef\nnop\n"); }
float fma1(float a, float b, float c) { return a * b + c; }
#elif defined(__riscv)
void dat(void) { __asm__ volatile("nop\n.word 0x12345678\nc.nop\n.half 0x0001\nnop\n"); }
long mixed(long a, long b) { return (a + 1) * (b - 3) + (a >> 2); }
#elif defined(__arm__)
void dat(void) { __asm__ volatile("nop\n.word 0x12345678\n.word 0xdeadbeef\nnop\n"); }
unsigned lit(void) { unsigned r; __asm__ volatile("ldr %0, =0xcafef00d\nb 1f\n.ltorg\n1:\n" : "=r"(r)); return r; }
int ldr2(int *p) { return p[0x400] + p[1]; }
#endif
