/* psprecomp — the smaller firmware libraries.
 *
 * Kernel_Library, UtilsForUser, StdioForUser, sceSuspendForUser,
 * LoadExecForUser, ModuleMgrForUser, sceCtrl and sceAudio. Individually small,
 * but collectively they are what a game's C runtime needs before main() gets
 * anywhere -- newlib's reentrancy setup alone wants interrupt masking, a
 * clock, and the standard file descriptors.
 */

#include "psprecomp/hle.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

/* ---- Kernel_Library ------------------------------------------------------
 * Interrupt masking. The mask belongs to the *thread* that opened the critical
 * section, not to the machine: with several guest threads on one CPU, a global
 * flag would let one thread's suspend silence interrupts inside another
 * thread's unrelated bracket. The scheduler owns that state, because it is
 * also what decides whether an interrupt may be delivered at a park. */

static void hle_CpuSuspendIntr(void) {
    psp_ret((uint32_t)psp_threadman_intr_suspend());   /* the cookie resume expects */
}

static void hle_CpuResumeIntr(void) {
    psp_threadman_intr_resume(psp_arg(0));
    psp_ret(SCE_KERNEL_ERROR_OK);
}

/* ---- UtilsForUser -------------------------------------------------------- */

/* Cache maintenance. There is no cache to write back -- the recompiled code
 * and the GE share one flat backing store -- so these are genuinely no-ops
 * rather than unimplemented. */
static void hle_CacheOp(void) { psp_ret(SCE_KERNEL_ERROR_OK); }

static void hle_LibcTime(void) {
    time_t t = time(NULL);
    uint32_t out = psp_arg(0);
    if (out) psp_write32(out, (uint32_t)t);
    psp_ret((uint32_t)t);
}

static void hle_LibcClock(void) {
    /* Microseconds since start. A game that uses this for frame pacing needs
     * it to advance, so it is derived from the host clock rather than being a
     * constant -- a frozen clock makes a game either spin or run at infinite
     * speed, both of which look like a hang. */
    psp_ret((uint32_t)((uint64_t)clock() * 1000000ull / CLOCKS_PER_SEC));
}

static void hle_LibcGettimeofday(void) {
    uint32_t tv = psp_arg(0);
    if (tv) {
        time_t t = time(NULL);
        psp_write32(tv, (uint32_t)t);
        psp_write32(tv + 4, 0);
    }
    psp_ret(SCE_KERNEL_ERROR_OK);
}

/* General-purpose I/O pins, wired to the debug board. Nothing is connected. */
static void hle_GetGPI(void) { psp_ret(0); }

/* ---- StdioForUser -------------------------------------------------------- */
/* These return the file descriptors, which sceIoWrite then recognises. */
static void hle_Stdin(void)  { psp_ret(0); }
static void hle_Stdout(void) { psp_ret(1); }
static void hle_Stderr(void) { psp_ret(2); }

/* ---- sceSuspendForUser --------------------------------------------------- */
/* Power management around suspend. Nothing suspends here. */
static void hle_ok(void) { psp_ret(SCE_KERNEL_ERROR_OK); }

/* ---- LoadExecForUser ----------------------------------------------------- */

static int g_exit_requested;
int psp_exit_requested(void) { return g_exit_requested; }

static void hle_ExitGame(void) {
    /* A game calling this is finished. Recording it rather than terminating
     * the process lets the host report what happened and dump its counters. */
    g_exit_requested = 1;
    fprintf(stderr, "psprecomp: the game called sceKernelExitGame\n");
    psp_ret(SCE_KERNEL_ERROR_OK);
}

static void hle_RegisterExitCallback(void) { psp_ret(SCE_KERNEL_ERROR_OK); }

/* ---- ModuleMgrForUser ---------------------------------------------------- */
/* A game-sharing microgame is self-contained and does not load further
 * modules, so these report a plausible identity rather than doing anything.
 * A title that genuinely loads PRXs at run time will need real ones. */
/* These report the id of the one loaded module.
 *
 * A previous note here recorded the opposite -- that returning an id was a
 * "plausible lie" and an error was the truthful answer -- on the strength of a
 * test showing byte-identical output either way. **That test was run while
 * `jal` never assigned `$ra`**, so every non-leaf function in the program was
 * returning through a stale register. A falsification obtained under broken
 * codegen is not a falsification.
 *
 * Re-run after that fix, the two answers differ clearly: returning an id makes
 * the game's own "libc:_getmodreent: no reent structure" diagnostic disappear
 * and drops bad memory accesses from 3 to 0.
 *
 * And an id is the *truthful* answer. The question is "which module owns this
 * address", a self-contained microgame is exactly one module, and the host has
 * loaded it. Reporting 1 states that; reporting UNKNOWN_MODULE denies a module
 * that demonstrably exists. */
#define SCE_KERNEL_ERROR_UNKNOWN_MODULE 0x80020139u
#define PSP_MAIN_MODULE_ID 1u

static void hle_GetModuleId(void)          { psp_ret(PSP_MAIN_MODULE_ID); }
static void hle_GetModuleIdByAddress(void) { psp_ret(PSP_MAIN_MODULE_ID); }
static void hle_ModuleOk(void)             { psp_ret(SCE_KERNEL_ERROR_OK); }

/* ---- sceCtrl ------------------------------------------------------------- */

static uint32_t g_buttons;
static uint8_t  g_analog_x = 128, g_analog_y = 128;
static uint32_t g_ctrl_frame;

void psp_ctrl_set(uint32_t buttons, uint8_t ax, uint8_t ay) {
    g_buttons = buttons;
    g_analog_x = ax;
    g_analog_y = ay;
}

static void hle_CtrlSet(void) { psp_ret(SCE_KERNEL_ERROR_OK); }

/* SceCtrlData: u32 timestamp, u32 buttons, u8 lx, u8 ly, then padding to 16. */
static void hle_ReadBufferPositive(void) {
    uint32_t buf = psp_arg(0), count = psp_arg(1);
    if (!count) count = 1;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t at = buf + i * 16;
        psp_write32(at, g_ctrl_frame++);
        psp_write32(at + 4, g_buttons);
        psp_write8(at + 8, g_analog_x);
        psp_write8(at + 9, g_analog_y);
        for (int k = 10; k < 16; k++) psp_write8(at + (uint32_t)k, 0);
    }
    psp_ret(count);
}

/* ---- sceAudio ------------------------------------------------------------ */

#define AUDIO_CHANNELS 8
/* The hardware's sample rate is fixed at 44.1 kHz: a buffer of N samples lasts
 * N/44100 s whatever the channel's format is (mono or stereo changes how many
 * bytes a sample takes, not how long it plays). */
#define AUDIO_RATE     44100u

typedef struct {
    int      reserved;
    uint32_t samples;     /* the buffer length the guest asked for */
    uint32_t format;
    int      playing;     /* a buffer is in flight on this channel */
    /* When the buffer in flight finishes, in the guest's own clock. Held as the
     * end rather than the start on purpose: a guest that resizes the buffer
     * while one is playing changes the *next* block, not the one already in the
     * hardware. */
    uint64_t end_us;
} audio_ch;
static audio_ch g_audio[AUDIO_CHANNELS];
static uint64_t g_audio_blocks;

uint64_t psp_audio_blocks(void) { return g_audio_blocks; }

/* How long a block on this channel plays for. */
static uint64_t buffer_us(const audio_ch *c) {
    return c->samples ? (uint64_t)c->samples * 1000000ull / AUDIO_RATE : 0;
}

static void hle_ChReserve(void) {
    /* (channel, samplecount, format) -- channel -1 means "any". */
    int32_t ch = (int32_t)psp_arg(0);
    if (ch < 0) {
        for (int i = 0; i < AUDIO_CHANNELS; i++)
            if (!g_audio[i].reserved) { ch = i; break; }
    }
    if (ch < 0 || ch >= AUDIO_CHANNELS) { psp_ret(0x80000001); return; }
    g_audio[ch].reserved = 1;
    g_audio[ch].samples = psp_arg(1);
    g_audio[ch].format  = psp_arg(2);
    g_audio[ch].playing = 0;
    g_audio[ch].end_us  = 0;
    psp_ret((uint32_t)ch);
}

static void hle_ChRelease(void) {
    int32_t ch = (int32_t)psp_arg(0);
    if (ch < 0 || ch >= AUDIO_CHANNELS) { psp_ret(0x80000001); return; }
    g_audio[ch].reserved = 0;
    g_audio[ch].playing = 0;
    psp_ret(SCE_KERNEL_ERROR_OK);
}

static void hle_SetChannelDataLen(void) {
    /* (channel, samplecount). A guest that resizes its buffer changes how long a
     * block plays for, so the pacing follows it rather than a value remembered
     * from the reserve. */
    int32_t ch = (int32_t)psp_arg(0);
    if (ch < 0 || ch >= AUDIO_CHANNELS || !g_audio[ch].reserved) { psp_ret(0x80000001); return; }
    g_audio[ch].samples = psp_arg(1);
    psp_ret(SCE_KERNEL_ERROR_OK);
}

/* Hand a buffer to the channel and, for the blocking variants, wait out the one
 * already in flight.
 *
 * That wait is the whole point of this section. On hardware the audio hardware
 * drains a buffer in real time, so `sceAudioOutput*Blocking` is what paces a
 * game's audio thread -- one call per buffer, never faster. A toolkit that
 * returns immediately turns that thread into a spin, and because the CPU is
 * only handed over where the guest waits, the spin starves every other thread.
 * Measured on LocoRoco before this: 411 869 907 iterations of the one loop
 * around this call, 12 parks and 27 ms of virtual time in a 30-second run, with
 * 11 threads alive and none of them getting anywhere.
 *
 * The wait is the buffer's own duration, taken from the guest's clock rather
 * than from the host's: the virtual clock is the one the guest reads, and the
 * host's own wait is capped elsewhere (REAL_TICK_US), so a game can be paced at
 * 44.1 kHz without the run taking 44.1 kHz of wall clock to get through.
 *
 * The argument layouts differ between the variants (OutputBlocking has no pan
 * pair), which is why only the channel is read here: the buffer pointer and the
 * volumes are not part of the timing. */
static uint32_t start_block(int32_t ch, int blocking) {
    if (ch < 0 || ch >= AUDIO_CHANNELS || !g_audio[ch].reserved) return 0x80000001;
    audio_ch *c = &g_audio[ch];
    uint64_t dur = buffer_us(c);
    if (blocking && c->playing) {
        uint64_t now = psp_threadman_vtime_us();
        if (now < c->end_us) psp_threadman_park_us(c->end_us - now);
    }
    c->end_us = psp_threadman_vtime_us() + dur;
    c->playing = 1;
    g_audio_blocks++;
    return c->samples;
}

static void hle_OutputBlocking(void) { psp_ret(start_block((int32_t)psp_arg(0), 1)); }

/* The non-blocking variant hands the buffer over and returns: a game that uses
 * it is deliberately not paced by it, and it is the *next* blocking call that
 * waits. */
static void hle_Output(void) { psp_ret(start_block((int32_t)psp_arg(0), 0)); }

/* What is left of the buffer playing now, in samples -- derived from the clock
 * and the buffer's length, not remembered. Zero means the channel is free,
 * which is what a game's audio loop reads to decide it can hand over more. */
static void hle_GetChannelRestLength(void) {
    int32_t ch = (int32_t)psp_arg(0);
    if (ch < 0 || ch >= AUDIO_CHANNELS || !g_audio[ch].reserved) { psp_ret(0x80000001); return; }
    audio_ch *c = &g_audio[ch];
    uint64_t now = psp_threadman_vtime_us();
    if (!c->playing || now >= c->end_us) { psp_ret(0); return; }
    psp_ret((uint32_t)((c->end_us - now) * AUDIO_RATE / 1000000ull));
}

void psp_misc_reset(void) {
    /* The interrupt mask is per thread and lives in the scheduler; a reset
     * here cannot reach another thread's bracket, and should not. */
    g_exit_requested = 0;
    g_buttons = 0;
    g_analog_x = g_analog_y = 128;
    g_ctrl_frame = 0;
    memset(g_audio, 0, sizeof g_audio);
    g_audio_blocks = 0;
}

void psp_misc_init(void) { psp_misc_reset(); }

void psp_misc_register(void) {
    psp_hle_register(0x092968F4, "Kernel_Library", "sceKernelCpuSuspendIntr", hle_CpuSuspendIntr);
    psp_hle_register(0x5F10D406, "Kernel_Library", "sceKernelCpuResumeIntr",  hle_CpuResumeIntr);

    psp_hle_register(0x79D1C3FA, "UtilsForUser", "sceKernelDcacheWritebackAll",           hle_CacheOp);
    psp_hle_register(0xB435DEC5, "UtilsForUser", "sceKernelDcacheWritebackInvalidateAll", hle_CacheOp);
    psp_hle_register(0x27CC57F0, "UtilsForUser", "sceKernelLibcTime",         hle_LibcTime);
    psp_hle_register(0x91E4F6A7, "UtilsForUser", "sceKernelLibcClock",        hle_LibcClock);
    psp_hle_register(0x71EC4271, "UtilsForUser", "sceKernelLibcGettimeofday", hle_LibcGettimeofday);
    psp_hle_register(0x37FB5C42, "UtilsForUser", "sceKernelGetGPI",           hle_GetGPI);

    psp_hle_register(0x172D316E, "StdioForUser", "sceKernelStdin",  hle_Stdin);
    psp_hle_register(0xA6BAB2E9, "StdioForUser", "sceKernelStdout", hle_Stdout);
    psp_hle_register(0xF78BA90A, "StdioForUser", "sceKernelStderr", hle_Stderr);

    psp_hle_register(0xEADB1BD7, "sceSuspendForUser", "sceKernelPowerLock",   hle_ok);
    psp_hle_register(0x3AEE7261, "sceSuspendForUser", "sceKernelPowerUnlock", hle_ok);
    psp_hle_register(0x090CCB3F, "sceSuspendForUser", "sceKernelPowerTick",   hle_ok);

    psp_hle_register(0x05572A5F, "LoadExecForUser", "sceKernelExitGame",             hle_ExitGame);
    psp_hle_register(0x4AC57943, "LoadExecForUser", "sceKernelRegisterExitCallback", hle_RegisterExitCallback);

    /* Observed but unidentified. The game calls this three times on its
     * heap-setup path, and an unimplemented call returns 0 -- which for a
     * module query means "no module" and makes heap establishment fail.
     * Eighteen plausible ModuleMgr names were tried against SHA-1 with no
     * match, so the name is genuinely unknown and is not invented here. */
    psp_hle_register_unnamed(0xF9275D98, "ModuleMgrForUser", hle_GetModuleId);
    psp_hle_register(0xF0A26395, "ModuleMgrForUser", "sceKernelGetModuleId",          hle_GetModuleId);
    psp_hle_register(0xD8B73127, "ModuleMgrForUser", "sceKernelGetModuleIdByAddress", hle_GetModuleIdByAddress);
    psp_hle_register(0x50F0C1EC, "ModuleMgrForUser", "sceKernelStartModule",          hle_ModuleOk);
    psp_hle_register(0xD1FF982A, "ModuleMgrForUser", "sceKernelStopModule",           hle_ModuleOk);
    psp_hle_register(0x2E0911AA, "ModuleMgrForUser", "sceKernelUnloadModule",         hle_ModuleOk);

    psp_hle_register(0x1F4011E6, "sceCtrl", "sceCtrlSetSamplingMode",     hle_CtrlSet);
    psp_hle_register(0x6A2774F3, "sceCtrl", "sceCtrlSetSamplingCycle",    hle_CtrlSet);
    psp_hle_register(0x1F803938, "sceCtrl", "sceCtrlReadBufferPositive",  hle_ReadBufferPositive);

    psp_hle_register(0x5EC81C55, "sceAudio", "sceAudioChReserve",            hle_ChReserve);
    psp_hle_register(0x6FC46853, "sceAudio", "sceAudioChRelease",            hle_ChRelease);
    psp_hle_register(0x136CAF51, "sceAudio", "sceAudioOutputBlocking",       hle_OutputBlocking);
    psp_hle_register(0x13F592BC, "sceAudio", "sceAudioOutputPannedBlocking", hle_OutputBlocking);
    psp_hle_register(0xE2D56B2D, "sceAudio", "sceAudioOutputPanned",         hle_Output);
    psp_hle_register(0xB011922F, "sceAudio", "sceAudioGetChannelRestLength", hle_GetChannelRestLength);
    psp_hle_register(0xCB2E439E, "sceAudio", "sceAudioSetChannelDataLen",    hle_SetChannelDataLen);
    psp_hle_register(0x95FD0C2D, "sceAudio", "sceAudioChangeChannelConfig",  hle_ok);
    psp_hle_register(0xB7E1D8E7, "sceAudio", "sceAudioChangeChannelVolume",  hle_ok);
}
