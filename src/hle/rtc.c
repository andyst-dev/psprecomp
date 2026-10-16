/* psprecomp — sceRtc: the real-time clock.
 *
 * A PSP tick is one microsecond since 0001-01-01. That is the unit
 * `sceRtcGetTickResolution()` reports, and the unit every date conversion in
 * the library is built on, so it is registered here rather than left as a
 * comment: it makes the unit of `sceRtcGetCurrentTick` a fact the tests hold.
 *
 * The clock a game reads is the host's wall clock, sampled once, plus the
 * scheduler's virtual time -- the same shape the reference implementations use,
 * and the shape that matters: a value that *advances with the run* rather than
 * a constant. A frozen clock is not neutral. A title that seeds anything from
 * the clock, or measures an interval with it, gets one of two wrong behaviours
 * from a constant -- a seed that never changes, or a division by a zero-length
 * interval -- and both look like a defect in the game's own code.
 *
 * The epoch offset is arithmetic rather than a copied magic number:
 * 0001-01-01 to 1970-01-01 is 719162 days, and 719162 * 86400 = 62135596800
 * seconds. Any date conversion a game performs has to agree with that number,
 * so it is written out as the product that produces it.
 *
 * Only the one entry point is implemented, because it is the only one a module
 * has needed so far. GetCurrentClock and the conversion functions fill a
 * `ScePspDateTime` and take a timezone, which is a claim about local time this
 * toolkit has no basis for yet; they go in when a title imports them.
 */

#include "psprecomp/hle.h"

#include <time.h>

#define TICKS_PER_SECOND  1000000ull
#define RTC_EPOCH_SECONDS (719162ull * 86400ull)   /* 0001-01-01 -> 1970-01-01 */

static uint64_t g_base_ticks;    /* the wall clock at init, in ticks */
static uint64_t g_last_tick;     /* the newest tick handed to the guest */
static uint64_t g_ticks_served;

void psp_rtc_reset(void) {
    g_base_ticks = 0;
    g_last_tick = 0;
    g_ticks_served = 0;
}

void psp_rtc_init(void) {
    time_t now = time(NULL);
    psp_rtc_reset();
    /* A host clock before 1970 would make the base a negative offset; clamping
     * keeps the tick a plausible date instead of a number from before year 1. */
    g_base_ticks = ((uint64_t)(now > 0 ? now : 0) + RTC_EPOCH_SECONDS) * TICKS_PER_SECOND;
}

/* The tick as the guest would read it now. Not a wall clock reading: it also
 * carries the run's own progress, so a game that measures an interval measures
 * something that moves even when the host clock is stopped.
 *
 * Note what this is *not* used for: there is no ticking RTC interrupt, so a
 * title that sleeps until an alarm never wakes. Alarms are their own entry
 * points (sceRtcSetAlarmTick and friends) and are not implemented -- the same
 * "unbounded wait" rule as everywhere else applies when one is asked for. */
static uint64_t rtc_now(void) { return g_base_ticks + psp_threadman_vtime_us(); }

uint64_t psp_rtc_ticks(void) { return g_ticks_served; }
uint64_t psp_rtc_last_tick(void) { return g_last_tick; }

static void hle_RtcGetCurrentTick(void) {
    uint32_t out = psp_arg(0);
    uint64_t tick = rtc_now();

    g_last_tick = tick;
    g_ticks_served++;

    /* A zero pointer is common here -- a game asking for a tick it does not
     * want -- and it cannot be written through: address zero is mapped in this
     * toolkit (it is where a PRX links), so a blind write there would edit the
     * module's own image rather than fail. The firmware returns success either
     * way, so the value is written when there is somewhere to write it. */
    if (out) {
        psp_write32(out, (uint32_t)tick);
        psp_write32(out + 4, (uint32_t)(tick >> 32));
    }
    psp_ret(SCE_KERNEL_ERROR_OK);
}

static void hle_RtcGetTickResolution(void) {
    psp_ret((uint32_t)TICKS_PER_SECOND);
}

void psp_rtc_register(void) {
    /* NIDs are SHA-1(name)[0:4] little-endian, from pspsdk's sceRtc.S;
     * tests/test_hle.c verifies every pair registered here against the name. */
    psp_hle_register(0x3F7AD767, "sceRtc", "sceRtcGetCurrentTick",     hle_RtcGetCurrentTick);
    psp_hle_register(0xC41C2853, "sceRtc", "sceRtcGetTickResolution", hle_RtcGetTickResolution);
}
