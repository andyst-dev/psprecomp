/* Regression test for the intra-function merge opt-out.
 *
 * A function with a backward branch into its own body is one MIPS function.
 * Discovery used to promote that branch target to a separate entry, which
 * split the body and made its fall-through run a frame it never pushed. The
 * merge keeps it inside; PSPPRECOMP_NO_INTRA_MERGE=1 restores the old split for
 * bisection. The value matters: =0 must keep the merge on.
 *
 * Synthetic code only, no game data. */

#include "analyze.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BASE 0x08804000u

static uint8_t image[0x100];
static int failures;

#define CHECK(cond, ...)                                        \
    do {                                                        \
        if (!(cond)) {                                          \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);         \
            printf(__VA_ARGS__);                                \
            printf("\n");                                       \
            failures++;                                         \
        }                                                       \
    } while (0)

static void put32(uint32_t addr, uint32_t word) {
    uint8_t *p = image + (addr - BASE);
    p[0] = (uint8_t)word; p[1] = (uint8_t)(word >> 8);
    p[2] = (uint8_t)(word >> 16); p[3] = (uint8_t)(word >> 24);
}

static void set_flag(const char *value) {
#ifdef _MSC_VER
    _putenv_s("PSPRECOMP_NO_INTRA_MERGE", value ? value : "");
#else
    if (value) setenv("PSPRECOMP_NO_INTRA_MERGE", value, 1);
    else       unsetenv("PSPRECOMP_NO_INTRA_MERGE");
#endif
}

/* Returns the function count for one setting of the opt-out. */
static int discover_count(const char *flag) {
    set_flag(flag);
    memset(image, 0, sizeof image);
    put32(BASE + 0x00, 0x24020001u);   /* addiu $v0, $zero, 1  */
    put32(BASE + 0x04, 0x00000000u);   /* nop                 */
    put32(BASE + 0x08, 0x10000001u);   /* beq   $zero,$zero, BASE+0x10 */
    put32(BASE + 0x0C, 0x00000000u);   /* nop                 */
    put32(BASE + 0x10, 0x24030002u);   /* addiu $v1, $zero, 2  */
    put32(BASE + 0x14, 0x1000FFFCu);   /* beq   $zero,$zero, BASE+0x08 (backward) */
    put32(BASE + 0x18, 0x00000000u);   /* nop                 */
    put32(BASE + 0x1C, 0x03E00008u);   /* jr    $ra            */
    put32(BASE + 0x20, 0x00000000u);   /* nop                 */

    a_analysis an;
    memset(&an, 0, sizeof an);
    an.code = image;
    an.base = BASE;
    an.size = 0x24;
    uint32_t seed = BASE;
    int rc = a_discover(&an, &seed, 1);
    int n = an.nfuncs;
    CHECK(rc == 0, "discovery succeeds with %s", flag ? flag : "no flag");
    a_analysis_free(&an);
    return n;
}

int main(void) {
    const int unset   = discover_count(NULL);
    const int zero    = discover_count("0");
    const int disabled = discover_count("1");
    set_flag(NULL);

    CHECK(unset == 1, "unset: the body stays one function, got %d", unset);
    CHECK(zero == 1, "=0 keeps the merge on, got %d functions", zero);
    CHECK(disabled == 2, "=1 restores the old split, got %d functions", disabled);

    if (failures) {
        printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    printf("all intra-merge checks passed (synthetic code, no game data)\n");
    return 0;
}
