/* Emitter tests — synthetic code only, no game data.
 *
 * The function under test is eight instructions of hand-assembled MIPS chosen
 * to exercise the parts of emission that are actually hard: a conditional
 * branch with a delay slot, a backward-reachable label, and a `jr $ra` whose
 * delay slot must run before the return.
 *
 * The assertions are about the *shape* of the generated C, because that shape
 * is the contract. In particular the branch must read its condition into a
 * temporary before the delay slot runs — if that ever regresses, a delay slot
 * that writes a condition register silently changes which way the branch goes,
 * once in a thousand iterations, in a game nobody can debug.
 */

#include "analyze.h"
#include "emit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

#define CHECK(cond, ...)                                       \
    do {                                                       \
        if (!(cond)) {                                         \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);        \
            printf(__VA_ARGS__);                               \
            printf("\n");                                      \
            failures++;                                        \
        }                                                      \
    } while (0)

#define BASE 0x08804000u

/* addiu $sp,$sp,-16      prologue
 * sw    $ra,12($sp)
 * addu  $v0,$a0,$a1
 * beq   $v0,$zero,+2     -> the `jr $ra` at BASE+24
 * addiu $v0,$v0,1        delay slot: runs either way, and writes $v0 which
 *                        the branch above just read
 * lw    $ra,12($sp)
 * jr    $ra
 * addiu $sp,$sp,16       delay slot: runs before the return
 */
static const uint32_t CODE[] = {
    0x27BDFFF0u, 0xAFBF000Cu, 0x00851021u, 0x10400002u,
    0x24420001u, 0x8FBF000Cu, 0x03E00008u, 0x27BD0010u,
};

static char *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    char *b = (char *)malloc((size_t)n + 1);
    if (!b) { fclose(f); return NULL; }
    size_t got = fread(b, 1, (size_t)n, f);
    b[got] = '\0';
    fclose(f);
    if (len) *len = got;
    return b;
}

/* Assert a substring appears, reporting what was missing if not. */
static void expect_contains(const char *hay, const char *needle, const char *why) {
    if (!strstr(hay, needle)) {
        printf("FAIL %s\n  expected to find: %s\n", why, needle);
        failures++;
    }
}

/* Transfers that END a body must be emitted so that the call is the LAST thing
 * the body does. All three of them -- a fall-out into a gap owned by another
 * function, a `j` to a function entry, the taken path of a likely branch --
 * compile to a call followed by `return`, and the C compiler turns that into a
 * sibling jump only when nothing sits in between. A stack observation placed
 * after the call costs a host frame per guest transfer instead, so a loop the
 * hardware runs without ever touching its stack consumes the host stack. The
 * ordinary-branch and end-of-body paths already emitted bare calls; these
 * checks exist so the other three sites cannot drift back.
 */

/* A gap: A branches forward over 0x08804008, which is B's entry, so the
 * sequential walk of A reaches an address owned by another function and stops
 * there. A's body therefore ends by transferring into B.
 *
 *   08804000  beq  $v0,$zero,0x08804010     A
 *   08804004  addiu $v0,$v0,1               delay slot
 *   08804008  jr   $ra                      B
 *   0880400C  nop
 *   08804010  jr   $ra                      A's other path
 *   08804014  nop
 */
static const uint32_t GAP_CODE[] = {
    0x10400003u, 0x24420001u, 0x03E00008u, 0x00000000u,
    0x03E00008u, 0x00000000u,
};

/* A `j` out of the body, to a function entry: A transfers at 0x08804000 and
 * never runs 0x08804008.
 *
 *   08804000  j    0x08804010               A
 *   08804004  lui  $v0,0x1234               delay slot
 *   08804008  nop
 *   0880400C  nop
 *   08804010  jr   $ra                      B
 *   08804014  nop
 */
static const uint32_t JUMP_CODE[] = {
    0x0A201004u, 0x3C021234u, 0x00000000u, 0x00000000u,
    0x03E00008u, 0x00000000u,
};

/* A likely branch out of the body: the taken path transfers to a function
 * entry, and its delay slot belongs to that path.
 *
 *   08804000  bltzl $v0,0x08804020          A
 *   08804004  addiu $v0,$v0,1               delay slot, taken path
 *   08804008  jr    $ra                     not-taken path
 *   0880400C  nop
 *   08804010  nop
 *   08804014  nop
 *   08804018  nop
 *   0880401C  nop
 *   08804020  jr    $ra                     B
 *   08804024  nop
 */
static const uint32_t LIKELY_CODE[] = {
    0x58400007u, 0x24420001u, 0x03E00008u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x03E00008u, 0x00000000u,
};

static uint8_t synth[sizeof LIKELY_CODE * 4];

/* A branch whose delay slot is another function's entry.
 *
 * Discovery promoted the slot address to an entry of its own -- a relocation
 * pointer or a function pointer in a table, which on a real module disagrees
 * with the analysis often (7 525 .sym boundaries for a game that analysis alone
 * splits into 3 637 functions). So this body does not own the slot and cannot
 * emit it inline; the walk reaches it by falling through instead, which covers
 * the NOT-taken path only. A branch delays BOTH paths, so the taken path was
 * jumping straight past an instruction the hardware always runs.
 *
 *   08804000  beq  $v0,$zero,0x08804010    A -- stays inside A
 *   08804004  addiu $v0,$v0,1              delay slot -- B's entry
 *   08804008  jr   $ra                     B
 *   0880400C  nop
 *   08804010  nop                          A's other path
 *   08804014  jr   $ra                     A
 *   08804018  nop
 */
static const uint32_t SLOT_GOTO_CODE[] = {
    0x10400003u, 0x24420001u, 0x03E00008u, 0x00000000u,
    0x00000000u, 0x03E00008u, 0x00000000u,
};

/* The same, but the branch leaves the body: the slot has to run before the
 * tail call.
 *
 *   08804000  beq  $v0,$zero,0x0880400C    A -- transfers out
 *   08804004  lui  $v0,0x1234              delay slot -- B's entry
 *   08804008  nop                          B
 *   0880400C  jr   $ra                     the target
 *   08804010  nop
 */
static const uint32_t SLOT_TAIL_CODE[] = {
    0x10400002u, 0x3C021234u, 0x00000000u, 0x03E00008u, 0x00000000u,
};

/* A likely branch whose delay slot belongs to another body. A likely branch
 * nullifies its slot when NOT taken, so the slot belongs inside the taken path
 * and nowhere else -- and with the slot owned by another body there was no
 * taken path to put it in.
 *
 *   08804000  blezl $v0,0x08804010         A
 *   08804004  addiu $v0,$v0,1              delay slot -- B's entry
 *   08804008  jr   $ra                     B
 *   0880400C  nop
 *   08804010  nop                          A's other path
 *   08804014  jr   $ra                     A
 *   08804018  nop
 */
static const uint32_t SLOT_LIKELY_CODE[] = {
    0x58400003u, 0x24420001u, 0x03E00008u, 0x00000000u,
    0x00000000u, 0x03E00008u, 0x00000000u,
};

/* Discover from `seeds` and emit, returning the generated .c (caller frees). */
static char *emit_synthetic(const char *prefix, const uint32_t *words, size_t nwords,
                            const uint32_t *seeds, int nseeds, int *nfuncs_out) {
    for (size_t i = 0; i < nwords; i++) {
        synth[i * 4 + 0] = (uint8_t)(words[i]);
        synth[i * 4 + 1] = (uint8_t)(words[i] >> 8);
        synth[i * 4 + 2] = (uint8_t)(words[i] >> 16);
        synth[i * 4 + 3] = (uint8_t)(words[i] >> 24);
    }

    a_analysis an;
    memset(&an, 0, sizeof an);
    an.code = synth;
    an.base = BASE;
    an.size = (uint32_t)(nwords * 4);

    if (a_discover(&an, (uint32_t *)seeds, nseeds) != 0) {
        printf("FAIL discovery failed for %s\n", prefix);
        failures++;
        a_analysis_free(&an);
        return NULL;
    }
    if (nfuncs_out) *nfuncs_out = an.nfuncs;

    emit_opts o;
    o.outdir = ".";
    o.prefix = prefix;
    o.module = "synthetic";
    int rc = a_emit(&an, &o);
    a_analysis_free(&an);
    if (rc != 0) {
        printf("FAIL emission failed for %s\n", prefix);
        failures++;
        return NULL;
    }

    char path[128];
    snprintf(path, sizeof path, "./%s_funcs.c", prefix);
    return slurp(path, NULL);
}

/* The transfer must END the body: skip the whitespace that follows the call and
 * accept only `return;` or the closing brace. Anything else costs a frame. */
static void expect_tail_transfer(const char *src, const char *call, const char *why) {
    const char *p = strstr(src, call);
    if (!p) { printf("FAIL %s\n  no such call emitted: %s\n", why, call); failures++; return; }
    const char *q = p + strlen(call);
    while (*q == ' ' || *q == '\t' || *q == '\n') q++;
    if (strncmp(q, "return;", 7) != 0 && *q != '}') {
        printf("FAIL %s\n  a statement follows the transfer, so it cannot stay a "
               "tail jump: %.40s\n", why, q);
        failures++;
    }
}

static void expect_absent(const char *src, const char *needle, const char *why) {
    if (strstr(src, needle)) {
        printf("FAIL %s\n  must not be emitted here: %s\n", why, needle);
        failures++;
    }
}

/* One body, from its `static void psp_body_<addr>` to the wrapper emitted right
 * after it.
 *
 * These checks have to be scoped this way. The delay slot is the *entry* of the
 * body next door, so the very same instruction is emitted twice in the file:
 * once in the taken path here, once as that body's first statement. An
 * unbounded `strstr` finds the neighbour's copy and passes for the wrong
 * reason -- measured: without the scoping, the tail-call case passed on the
 * rule that had no slot in the taken path at all. */
static const char *find_in(const char *from, const char *to, const char *needle) {
    const char *p = strstr(from, needle);
    return (p && p < to) ? p : NULL;
}

static char *body_window(const char *src, const char *name, size_t *len) {
    const char *b = strstr(src, name);
    if (!b) return NULL;
    const char *e = strstr(b, "\nvoid psp_func_");
    if (!e) e = b + strlen(b);
    *len = (size_t)(e - b);
    char *w = (char *)malloc(*len + 1);
    if (!w) return NULL;
    memcpy(w, b, *len);
    w[*len] = '\0';
    return w;
}

static void check_terminal_transfer_stays_tail(void) {
    /* --- a gap owned by another function --- */
    {
        uint32_t seeds[2] = { BASE, BASE + 8u };
        int n = 0;
        char *src = emit_synthetic("t_gap", GAP_CODE, 6, seeds, 2, &n);
        CHECK(n == 2, "the gap block discovers two entries, got %d", n);
        if (src) {
            expect_tail_transfer(src, "psp_func_08804008();",
                                 "a fall-out into another function stays a tail jump");
            expect_absent(src, "PSP_SP_CALL(0x08804008u",
                          "no stack observation after the fall-out transfer");
            /* Both labels are 0x08804008 -- the fall-out names its target, and
             * that target's own `jr $ra` names itself -- so the assertion has to
             * be scoped to the gap: the judgement between the comment and the
             * transfer must be the fall-out one. */
            const char *gap = strstr(src, "/* falls into 0x08804008, owned by another function */");
            const char *call = strstr(src, "psp_func_08804008();");
            const char *leave = gap ? strstr(gap, "PSP_SP_LEAVE(0x08804008u);") : NULL;
            const char *check = gap ? strstr(gap, "PSP_SP_CHECK(0x08804008u);") : NULL;
            CHECK(gap && leave && call && gap < leave && leave < call,
                  "the fall-out judges a hand-over, before the transfer");
            CHECK(!check || check > call,
                  "a fall-out is not judged as if it were a return");
            free(src);
        }
    }

    /* --- a `j` to a function entry --- */
    {
        uint32_t seeds[2] = { BASE, BASE + 0x10u };
        int n = 0;
        char *src = emit_synthetic("t_jump", JUMP_CODE, 6, seeds, 2, &n);
        CHECK(n == 2, "the jump block discovers two entries, got %d", n);
        if (src) {
            expect_tail_transfer(src, "psp_func_08804010();",
                                 "a `j` to a function entry stays a tail jump");
            expect_absent(src, "PSP_SP_CALL(0x08804010u",
                          "no stack observation after the `j` transfer");
            free(src);
        }
    }

    /* --- the taken path of a likely branch --- */
    {
        uint32_t seeds[2] = { BASE, BASE + 0x20u };
        int n = 0;
        char *src = emit_synthetic("t_likely", LIKELY_CODE, 10, seeds, 2, &n);
        CHECK(n == 2, "the likely block discovers two entries, got %d", n);
        if (src) {
            expect_tail_transfer(src, "psp_func_08804020();",
                                 "the taken path of a likely branch stays a tail jump");
            expect_absent(src, "PSP_SP_CALL(0x08804020u",
                          "no stack observation on the likely branch's taken path");
            free(src);
        }
    }
}

/* A branch delays BOTH paths. A slot this body owns is emitted inline and runs
 * on both; a slot another body owns cannot be, so the taken path has to run it
 * itself -- otherwise the branch jumps over an instruction the hardware always
 * executes, and the code at the target reads a register nobody set.
 *
 * Measured on LocoRoco (UCES00304): 0x1571C is `beq $v0,$zero,0x15774` whose
 * slot at 0x15720 is `lui $v0,0x1C`, and 0x15774 loads through $v0; the taken
 * path left it zero and the game read the module's own code as a table. */
static void check_delay_slot_runs_on_the_taken_path(void) {
    /* --- the branch stays inside its body --- */
    {
        uint32_t seeds[2] = { BASE, BASE + 4u };
        int n = 0;
        char *src = emit_synthetic("t_slot_goto", SLOT_GOTO_CODE, 7, seeds, 2, &n);
        size_t len = 0;
        char *body = src ? body_window(src, "static void psp_body_08804000", &len) : NULL;
        if (body) {
            const char *end = body + len;
            const char *cond = find_in(body, end, "int _c = (r_v0 == r_zero);");
            const char *take = cond ? find_in(cond, end, "if (_c) {") : NULL;
            const char *slot = take ? find_in(take, end, "r_v0 = r_v0 + 1;") : NULL;
            const char *jump = slot ? find_in(slot, end, "goto L_08804010;") : NULL;
            CHECK(cond && take && slot && jump,
                  "the taken path runs the delay slot before it branches inside");
            free(body);
        }
        free(src);
    }

    /* --- the branch leaves the body through a tail call --- */
    {
        uint32_t seeds[2] = { BASE, BASE + 4u };
        int n = 0;
        char *src = emit_synthetic("t_slot_tail", SLOT_TAIL_CODE, 5, seeds, 2, &n);
        size_t len = 0;
        char *body = src ? body_window(src, "static void psp_body_08804000", &len) : NULL;
        if (body) {
            const char *end = body + len;
            const char *cond = find_in(body, end, "int _c = (r_v0 == r_zero);");
            const char *take = cond ? find_in(cond, end, "if (_c) {") : NULL;
            const char *slot = take ? find_in(take, end, "r_v0 = 0x12340000u;") : NULL;
            const char *call = slot ? find_in(slot, end, "psp_func_0880400C();") : NULL;
            CHECK(cond && take && slot && call,
                  "the taken path runs the delay slot before the tail transfer");
            expect_tail_transfer(body, "psp_func_0880400C();",
                                 "the taken path of a branch still ends in the transfer");
            free(body);
        }
        free(src);
    }

    /* --- a likely branch: the slot belongs to the taken path and nowhere else --- */
    {
        uint32_t seeds[2] = { BASE, BASE + 4u };
        int n = 0;
        char *src = emit_synthetic("t_slot_likely", SLOT_LIKELY_CODE, 7, seeds, 2, &n);
        size_t len = 0;
        char *body = src ? body_window(src, "static void psp_body_08804000", &len) : NULL;
        if (body) {
            const char *end = body + len;
            const char *take = find_in(body, end, "if ((int32_t)r_v0 <= 0) {");
            const char *slot = take ? find_in(take, end, "r_v0 = r_v0 + 1;") : NULL;
            const char *jump = slot ? find_in(slot, end, "goto L_08804010;") : NULL;
            CHECK(take && slot && jump,
                  "a likely branch runs its delay slot inside the taken path");
            free(body);
        }
        free(src);
    }
}

int main(void) {
    uint8_t code[sizeof CODE];
    for (size_t i = 0; i < sizeof CODE / sizeof CODE[0]; i++) {
        code[i * 4 + 0] = (uint8_t)(CODE[i]);
        code[i * 4 + 1] = (uint8_t)(CODE[i] >> 8);
        code[i * 4 + 2] = (uint8_t)(CODE[i] >> 16);
        code[i * 4 + 3] = (uint8_t)(CODE[i] >> 24);
    }

    a_analysis an;
    memset(&an, 0, sizeof an);
    an.code = code;
    an.base = BASE;
    an.size = (uint32_t)sizeof code;

    uint32_t seed = BASE;
    CHECK(a_discover(&an, &seed, 1) == 0, "discovery runs");
    CHECK(an.nfuncs == 1, "one function found, got %d", an.nfuncs);
    if (an.nfuncs != 1) return 1;
    CHECK(an.funcs[0].addr == BASE, "function entry is the seed");
    CHECK(an.funcs[0].has_return, "function reaches a `jr $ra`");
    CHECK(an.insns == 8, "all eight instructions visited, got %llu",
          (unsigned long long)an.insns);

    emit_opts o;
    o.outdir = ".";
    o.prefix = "t_emit";
    o.module = "synthetic";
    CHECK(a_emit(&an, &o) == 0, "emission succeeds");

    char *src = slurp("./t_emit_funcs.c", NULL);
    CHECK(src != NULL, "generated .c is readable");
    if (!src) return 1;

    expect_contains(src, "void psp_func_08804000(void)",
                    "function is named after its address");

    /* The prologue, lowered to plain C. */
    expect_contains(src, "r_sp = r_sp + -16;", "addiu lowers to arithmetic");
    expect_contains(src, "psp_write32(r_sp + 12, r_ra);", "sw lowers to a memory write");
    expect_contains(src, "r_v0 = r_a0 + r_a1;", "addu lowers to addition");

    /* Every statement carries its address and disassembly. */
    expect_contains(src, "/* 08804000  addiu", "statements carry their disassembly");

    /* THE important one: the branch condition is captured into a temporary
     * *before* the delay slot executes, because the delay slot writes $v0 and
     * the hardware read the old value. */
    expect_contains(src, "int _c = (r_v0 == r_zero);",
                    "branch condition is captured before the delay slot");
    {
        const char *cond = strstr(src, "int _c = (r_v0 == r_zero);");
        const char *slot = strstr(src, "r_v0 = r_v0 + 1;");
        const char *jump = cond ? strstr(cond, "if (_c) goto") : NULL;
        CHECK(cond && slot && jump && cond < slot && slot < jump,
              "ordering must be condition -> delay slot -> branch");
    }

    /* The branch target is a label, and the return runs its delay slot first. */
    expect_contains(src, "L_08804018:", "branch target becomes a label");
    {
        const char *lbl = strstr(src, "L_08804018:");
        const char *slot = lbl ? strstr(lbl, "r_sp = r_sp + 16;") : NULL;
        const char *ret = slot ? strstr(slot, "return;") : NULL;
        CHECK(slot && ret, "the `jr $ra` delay slot is emitted before the return");
    }

    /* $zero is never assigned: `addu $v0,$a0,$a1` reads it, but nothing may
     * write it. A generated `r_zero =` would corrupt every later use. */
    CHECK(strstr(src, "r_zero =") == NULL, "nothing ever assigns $zero");

    free(src);

    /* The header declares the function and the registration entry point. */
    char *hdr = slurp("./t_emit_funcs.h", NULL);
    CHECK(hdr != NULL, "generated .h is readable");
    if (hdr) {
        expect_contains(hdr, "void psp_func_08804000(void);", "header declares the function");
        expect_contains(hdr, "void psp_recomp_register(void);", "header declares registration");
        expect_contains(hdr, "#define r_sp", "header defines register aliases");
        free(hdr);
    }

    /* Registration wires the address to the function. */
    src = slurp("./t_emit_funcs.c", NULL);
    if (src) {
        expect_contains(src, "psp_register(0x08804000u, psp_func_08804000);",
                        "the function is registered for indirect dispatch");
        free(src);
    }

    a_analysis_free(&an);

    /* ---- viim/vfim: destination register and non-finite immediates ---------
     * One real word from LocoRoco (UCES00304). Two separate defects used to
     * show up on it, so both are asserted here:
     *   - the destination came out as 1 where the encoding says 22, so the
     *     generated C wrote the wrong VFPU register;
     *   - the NaN halfword rendered as `nanf`, an invalid token that stopped
     *     the emitted file from compiling at all.
     */
    static const uint32_t CODE_VFIM[] = {
        0xDF967E81u,   /* vfim v22,  NaN halfword        */
        0xDFFF7C00u,   /* vfim v127, +Inf halfword       */
        0xDF96FC00u,   /* vfim v22,  -Inf halfword       */
        0xDF969B81u,   /* vfim v22,  -0.00366401672      */
        0x03E00008u,   /* jr $ra                         */
        0x00000000u,
    };
    uint8_t code2[sizeof CODE_VFIM];
    for (size_t i = 0; i < sizeof CODE_VFIM / sizeof CODE_VFIM[0]; i++) {
        code2[i * 4 + 0] = (uint8_t)(CODE_VFIM[i]);
        code2[i * 4 + 1] = (uint8_t)(CODE_VFIM[i] >> 8);
        code2[i * 4 + 2] = (uint8_t)(CODE_VFIM[i] >> 16);
        code2[i * 4 + 3] = (uint8_t)(CODE_VFIM[i] >> 24);
    }

    a_analysis an2;
    memset(&an2, 0, sizeof an2);
    an2.code = code2;
    an2.base = BASE;
    an2.size = (uint32_t)sizeof code2;

    uint32_t seed2 = BASE;
    CHECK(a_discover(&an2, &seed2, 1) == 0, "vfim discovery runs");

    emit_opts o2;
    o2.outdir = ".";
    o2.prefix = "t_vfim";
    o2.module = "synthetic";
    CHECK(a_emit(&an2, &o2) == 0, "vfim emission succeeds");

    char *vsrc = slurp("./t_vfim_funcs.c", NULL);
    CHECK(vsrc != NULL, "generated vfim .c is readable");
    if (vsrc) {
        /* The destination is the encoding's vt field, not the generic vd. */
        expect_contains(vsrc, "psp_vimm(22, ",
                        "the destination is the encoding's vt field (bits 16..22)");
        expect_contains(vsrc, "psp_vimm(127, ",
                        "a destination of 127 survives the round trip");
        /* The immediates: all three branches of the non-finite handling. */
        expect_contains(vsrc, "psp_vimm(22, NAN);",
                        "a NaN halfword is spelled with the C99 NAN macro");
        expect_contains(vsrc, "psp_vimm(127, INFINITY);",
                        "a +Inf halfword is spelled INFINITY");
        expect_contains(vsrc, "psp_vimm(22, -INFINITY);",
                        "a -Inf halfword keeps its sign");
        expect_contains(vsrc, "psp_vimm(22, -0.00366401672f);",
                        "a finite halfword is emitted as before");
        /* `%.9gf` glued the format's literal `f` onto printf's `nan` and `inf`
         * output, giving these two invalid tokens. Neither may come back. */
        CHECK(strstr(vsrc, "nanf") == NULL, "no invalid `nanf` token");
        CHECK(strstr(vsrc, "inff") == NULL, "no invalid `inff` token");
        free(vsrc);
    }
    a_analysis_free(&an2);

    check_terminal_transfer_stays_tail();
    check_delay_slot_runs_on_the_taken_path();

    if (failures) {
        printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    printf("all emitter checks passed (synthetic code, no game data)\n");
    return 0;
}
