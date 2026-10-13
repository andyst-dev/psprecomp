/* psprecomp — InterruptManager, and the one place an interrupt is delivered.
 *
 * LocoRoco (UCES00304) is what put this file here, and the measurement is worth
 * keeping, because the shape of it is not what the API suggests.
 *
 * Its host reached a loop that ran 2 900 000 000 times in 30 seconds:
 *
 *     while (psp_read32(0x008E4EC4) == 0)     // a flag in the game's own .bss
 *         sceKernelSleepThread();
 *     sceKernelCpuSuspendIntr();              // and a critical-section bracket
 *       ... shared counter ...                // which says what this really is
 *     sceKernelCpuResumeIntr();
 *
 * That flag is armed by a sub-interrupt handler. `sceKernelSleepThread` was
 * already implemented -- as an immediate return, deliberately (threadman.c
 * argued the trade-off) -- so the sleep was not what was missing. These three
 * functions were: the game's registration went nowhere, so no handler could
 * ever run, so the flag could never be armed, so the wait could never end. No
 * amount of implementing `sleep` gets a loop out of that, because the loop's
 * exit condition is set by an interrupt nobody delivers.
 *
 * ## What an interrupt means here
 *
 * There is no timer and no preemption, so there is nothing to interrupt *from*.
 * An interrupt is therefore delivered at the points where the guest itself asks
 * for time to pass: SleepThread, DelayThread, the vblank wait, and every tick a
 * blocked thread spends parked. One wait, one delivery, handlers in registration
 * order, enabled ones only.
 *
 * A critical section is respected: a thread that has called
 * sceKernelCpuSuspendIntr gets no delivery until it resumes, because the flag
 * the handlers arm is written inside one of those brackets. The mask is
 * per-thread and lives in the scheduler (src/hle/threadman.c).
 *
 * That is a coarse model, and it is the honest one. It is also the model that
 * lets a *polling* game advance, which the alternative cannot: returning from
 * the wait with nothing delivered leaves the thing being polled for unset,
 * forever.
 */
#include "psprecomp/hle.h"
#include "psprecomp/dispatch.h"

#include <stdio.h>
#include <string.h>

#define MAX_SUBINTR 64
#define INTR_LOG    8   /* how many problems to name before counting silently */

typedef struct {
    int      intr;
    int      subintr;
    uint32_t handler;
    uint32_t arg;
    int      enabled;
    int      used;
} psp_subintr;

static psp_subintr g_sub[MAX_SUBINTR];
static uint64_t   g_deliveries;
static uint64_t   g_unresolved;
static int        g_unresolved_logged;
static int        g_delivering;   /* reentrancy guard: see psp_intr_tick */

void psp_intrman_init(void) { psp_intrman_reset(); }

void psp_intrman_reset(void) {
    memset(g_sub, 0, sizeof g_sub);
    g_deliveries = 0;
    g_unresolved = 0;
    g_unresolved_logged = 0;
    g_delivering = 0;
}

static psp_subintr *find_subintr(int intr, int subintr) {
    for (int i = 0; i < MAX_SUBINTR; i++)
        if (g_sub[i].used && g_sub[i].intr == intr && g_sub[i].subintr == subintr)
            return &g_sub[i];
    return NULL;
}

static void hle_RegisterSubIntrHandler(void) {
    int      intr = (int)psp_arg(0);
    int      sub  = (int)psp_arg(1);
    uint32_t fn   = psp_arg(2);
    uint32_t arg  = psp_arg(3);

    /* Re-registering an (intr, subintr) pair replaces the handler, as it does on
     * hardware. Replacing also clears the enabled state: the guest has to enable
     * the new one, the same way it had to enable the old one. */
    psp_subintr *s = find_subintr(intr, sub);
    if (!s) {
        for (int i = 0; i < MAX_SUBINTR; i++)
            if (!g_sub[i].used) { s = &g_sub[i]; break; }
    }
    if (!s) { psp_ret(SCE_KERNEL_ERROR_NO_MEMORY); return; }

    s->intr = intr;
    s->subintr = sub;
    s->handler = fn;
    s->arg = arg;
    s->enabled = 0;
    s->used = 1;
    psp_ret(SCE_KERNEL_ERROR_OK);
}

/* Returns the previous enable state, 1 or 0, as the kernel does -- which also
 * makes it one of the rare calls whose zero return really is zero, not success. */
static void hle_EnableSubIntr(void) {
    psp_subintr *s = find_subintr((int)psp_arg(0), (int)psp_arg(1));
    if (!s) { psp_ret(SCE_KERNEL_ERROR_ERROR); return; }
    int was = s->enabled;
    s->enabled = 1;
    psp_ret((uint32_t)was);
}

static void hle_ReleaseSubIntrHandler(void) {
    psp_subintr *s = find_subintr((int)psp_arg(0), (int)psp_arg(1));
    if (!s) { psp_ret(SCE_KERNEL_ERROR_ERROR); return; }
    s->used = 0;
    psp_ret(SCE_KERNEL_ERROR_OK);
}

/* One interrupt, delivered to every enabled handler.
 *
 * Called from the wait points, so it runs *inside* a firmware call the guest
 * made, with the guest's live registers in `psp_cpu`. A real interrupt does not
 * disturb what it interrupted, so the whole register file is saved and restored
 * around each handler -- the same thing start_thread does when it enters a
 * thread body.
 *
 * The handler runs on the current $sp rather than a stack of its own. That is
 * correct for a call: a MIPS callee allocates *below* the caller's frame, so its
 * prologue cannot overwrite the frame it interrupted.
 */
void psp_intr_tick(void) {
    /* A handler is free to wait -- and handlers routinely do. The wait calls
     * back into here, so without this guard a single delivery would recurse
     * until the host stack ran out. An interrupt arriving while one is being
     * serviced is simply serviced later, which is what the flag being set
     * achieves anyway. */
    if (g_delivering) return;
    /* Inside a critical section, the guest has said that what it is doing
     * cannot be disturbed -- and what the handlers arm is written inside one
     * of those brackets. */
    if (psp_threadman_intr_disabled()) return;
    g_delivering = 1;

    for (int i = 0; i < MAX_SUBINTR; i++) {
        psp_subintr *s = &g_sub[i];
        if (!s->used || !s->enabled) continue;

        if (!psp_lookup(s->handler)) {
            /* Worth naming loudly: the guest registered a handler we cannot
             * call, which means the address is not in the recompiled set and
             * discovery missed a function. Silently skipping it would look
             * exactly like the bug this file exists to fix. */
            g_unresolved++;
            if (g_unresolved_logged < INTR_LOG) {
                g_unresolved_logged++;
                fprintf(stderr,
                        "psprecomp: sub-intr handler 0x%08X (intr %d, subintr %d)"
                        " is not a recompiled function\n",
                        s->handler, s->intr, s->subintr);
            }
            continue;
        }

        psp_cpu_state saved = psp_cpu;
        psp_cpu.r[PSP_REG_A0] = (uint32_t)s->subintr;   /* handler(subintr, arg) */
        psp_cpu.r[PSP_REG_A1] = s->arg;
        psp_cpu.r[PSP_REG_RA] = 0;
        psp_dispatch(s->handler);
        psp_cpu = saved;

        g_deliveries++;
    }

    g_delivering = 0;
}

int psp_intr_handlers(void) {
    int n = 0;
    for (int i = 0; i < MAX_SUBINTR; i++) if (g_sub[i].used) n++;
    return n;
}

int psp_intr_enabled(void) {
    int n = 0;
    for (int i = 0; i < MAX_SUBINTR; i++) if (g_sub[i].used && g_sub[i].enabled) n++;
    return n;
}

uint64_t psp_intr_deliveries(void) { return g_deliveries; }
uint64_t psp_intr_unresolved(void) { return g_unresolved; }

void psp_intrman_register(void) {
    /* NIDs are SHA-1(name)[0:4] little-endian; tests/test_hle.c verifies every
     * pair registered here. */
    psp_hle_register(0xCA04A2B9, "InterruptManager", "sceKernelRegisterSubIntrHandler", hle_RegisterSubIntrHandler);
    psp_hle_register(0xD61E6961, "InterruptManager", "sceKernelReleaseSubIntrHandler",  hle_ReleaseSubIntrHandler);
    psp_hle_register(0xFB8E22EC, "InterruptManager", "sceKernelEnableSubIntr",          hle_EnableSubIntr);
}
