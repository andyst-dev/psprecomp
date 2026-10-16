/* psprecomp — sceUmdUser: the drive a disc game boots from.
 *
 * A UMD title asks the drive whether there is a disc, waits for it to spin up,
 * and registers a callback for disc events. What a game actually consumes is
 * the *state* the firmware reports; the bytes it reads come through sceIo,
 * which this toolkit already serves from a host directory (`disc0:/` becomes
 * `<root>/disc/`, see iofilemgr.c). So what this file models is the state
 * machine, and nothing about the medium.
 *
 * **There is a disc in the drive, by construction.** The module under test was
 * loaded from one: a host that has a module to run has the medium it came from.
 * Reporting NOT_PRESENT would deny that, and a title that waits for a disc
 * before it starts would then be waiting for an event this toolkit cannot
 * produce -- which is indistinguishable from a hang. misc.c makes the same
 * argument for answering a module query with a module id rather than
 * UNKNOWN_MODULE.
 *
 * The bit values are pspsdk's `pspUmdState`:
 *
 *     0x01 not present   0x02 present   0x04 changed
 *     0x08 initing       0x10 inited    0x20 ready
 *
 * The public references disagree about the *names* of the two upper bits
 * (PPSSPP calls 0x10 READY and 0x20 READABLE) but not about the numbers, and
 * the numbers are the contract -- they are what a guest masks with. Masks seen
 * in a real title are 0x20, 0x30, 0x02, 0x10 and 0x0D. Both references report
 * the same two values for the two states this file has:
 *
 *     disc in, drive not activated    0x12   present | inited
 *     disc in, drive activated        0x32   present | inited | ready
 *
 * 0x32 answers every one of those masks the way an activated drive does --
 * 0x20, 0x30, 0x02 and 0x10 are all set -- and 0x0D, which asks "is the disc
 * absent, changed, or still spinning up", correctly gets no for an answer.
 */

#include "psprecomp/hle.h"

#include <stdio.h>
#include <string.h>

#define UMD_STAT_NOT_PRESENT 0x01u
#define UMD_STAT_PRESENT     0x02u
#define UMD_STAT_CHANGED     0x04u
#define UMD_STAT_INITING     0x08u
#define UMD_STAT_INITED      0x10u
#define UMD_STAT_READY       0x20u

#define UMD_STAT_INSERTED  (UMD_STAT_PRESENT | UMD_STAT_INITED)    /* 0x12 */
#define UMD_STAT_ACTIVATED (UMD_STAT_INSERTED | UMD_STAT_READY)    /* 0x32 */

/* Every state a drive can be in except CHANGED, which is an event rather than
 * a state and cannot be waited for. A wait naming anything outside this is not
 * asking about a drive at all, and the firmware answers it as a bad argument
 * instead of waiting for something it could never reach. */
#define UMD_STAT_WAITABLE \
    (UMD_STAT_NOT_PRESENT | UMD_STAT_PRESENT | UMD_STAT_INITING | \
     UMD_STAT_INITED | UMD_STAT_READY)

static uint32_t g_stat;
static uint32_t g_cbid;         /* the one registered drive callback, 0 for none */
static uint64_t g_unsatisfied;  /* waits this model could not satisfy */
static uint64_t g_notified;     /* state changes the registered callback was told about */
static uint64_t g_registrations;/* drive callbacks the guest registered (they can be replaced) */
static uint32_t g_reported;     /* which of those states have been named already */

void psp_umd_reset(void) {
    g_stat = UMD_STAT_INSERTED;
    g_cbid = 0;
    g_unsatisfied = 0;
    g_notified = 0;
    g_registrations = 0;
    g_reported = 0;
}

void psp_umd_init(void) { psp_umd_reset(); }

/* Non-zero means a wait asked for a state this drive never reports. That is the
 * one thing this subsystem can get wrong while looking healthy: the wait
 * returns, the game carries on, and nothing says the disc it was waiting for
 * was never going to arrive. Printed once per distinct state as well; the
 * counter is what a host's report shows for a run with no log at all. */
uint64_t psp_umd_unsatisfied_waits(void) { return g_unsatisfied; }

/* The state a guest would read right now. Worth showing in a host's report for
 * the same reason the counter above is: an unimplemented call returns 0, and a
 * drive stat of 0 reads as "no disc", which is a plausible answer a game acts
 * on rather than an error it can notice. */
uint32_t psp_umd_drive_stat(void) { return g_stat; }

/* How many state changes were handed to the registered callback. Like the two
 * above: a notification that was never raised and a notification that was
 * raised and never delivered look identical from outside, and this is the half
 * this file owns. */
uint64_t psp_umd_notifications(void) { return g_notified; }

/* How many times the guest registered one, so that "registered but never told"
 * is distinguishable from "never registered" in a report. Both look the same
 * from outside otherwise, and they mean opposite things: a title waiting on a
 * drive callback it did register is waiting on a state change, while one that
 * never registered has no reason to. */
uint64_t psp_umd_registrations(void) { return g_registrations; }

/* A drive state change is a notification to the game's callback, which is what
 * the callback is registered for. The argument is the state itself: that is what
 * the callback's own mask is built to read, and this title's callback tests
 * `arg & 0x3B` -- a copy of the state bits -- before anything else.
 *
 * The firmware's exact notification value is not sourced here (the reference
 * implementation could not be consulted while this was written), and the choice
 * matters less than it looks: any value carrying a state bit makes this game's
 * callback proceed, and the state it then reads is the same one GetDriveStat
 * would answer. What is stated rather than assumed is that the notification
 * happens on a *change* -- activation and deactivation, the two places this
 * model can change the state. */
static void notify_drive_state(void) {
    if (!g_cbid) return;
    if (psp_threadman_notify_callback(g_cbid, g_stat)) g_notified++;
}

/* ---- the calls ----------------------------------------------------------- */

static void hle_UmdActivate(void) {
    /* sceUmdActivate(unit, drive). The firmware takes unit 1 or 2 and a mount
     * point spelled `disc0:`, and mounts the drive there. Nothing is mounted
     * here -- sceIo rewrites `disc0:/` into the host root already, so the drive
     * has no mount duty left -- but the two arguments are still checked, because
     * a bad call has to be named here: the alternative is a game whose reads
     * quietly return nothing, far from this line. */
    int32_t unit = (int32_t)psp_arg(0);
    uint32_t name = psp_arg(1);
    char drive[16];

    if (unit != 1 && unit != 2) { psp_ret(SCE_KERNEL_ERROR_ILLEGAL_ARGUMENT); return; }
    if (!name) { psp_ret(SCE_KERNEL_ERROR_ILLEGAL_ARGUMENT); return; }
    psp_str(name, drive, sizeof drive);
    if (strcmp(drive, "disc0:") != 0) { psp_ret(SCE_KERNEL_ERROR_ILLEGAL_ARGUMENT); return; }

    g_stat = UMD_STAT_ACTIVATED;
    notify_drive_state();
    psp_ret(SCE_KERNEL_ERROR_OK);
}

static void hle_UmdDeactivate(void) {
    /* The firmware accepts a much wider range of units here than it does for
     * Activate, and does not compare the mount point (it is optional unless the
     * unit is 2). Reproduced as reported rather than tidied up: a check that is
     * wrong in the permissive direction costs nothing, and one that is wrong in
     * the strict direction rejects a call the firmware accepts. */
    int32_t unit = (int32_t)psp_arg(0);
    if (unit < 1 || unit > 18) { psp_ret(SCE_KERNEL_ERROR_ILLEGAL_ARGUMENT); return; }
    g_stat = UMD_STAT_INSERTED;
    notify_drive_state();
    psp_ret(SCE_KERNEL_ERROR_OK);
}

static void hle_UmdGetDriveStat(void) {
    psp_ret(g_stat);
}

static void hle_UmdWaitDriveStat(void) {
    uint32_t stat = psp_arg(0);

    if (!(stat & UMD_STAT_WAITABLE)) { psp_ret(SCE_KERNEL_ERROR_ILLEGAL_ARGUMENT); return; }
    if (stat & g_stat) { psp_ret(SCE_KERNEL_ERROR_OK); return; }

    /* The state is already there half the time and the faithful answer is
     * immediate, which is the common case and the one worth keeping exact.
     *
     * The other case is a wait this model cannot satisfy: the drive state is
     * fixed once the disc is in and the drive is activated, and nothing in this
     * toolkit can change it -- there is no insertion, no ejection, no read
     * error. Parking the thread until the state changes, which is what the
     * firmware does, would be a thread that never wakes: a run with no evidence
     * in it. The rule this toolkit applies to every unbounded wait is to cap it
     * and *name what was awaited*, so one scheduling point is given away (other
     * threads run), the request is counted, and the first time each distinct
     * state is asked for it is printed. A game that then misbehaves has the
     * reason in the log instead of a silence that reads like a defect in its
     * own code. */
    if (!(g_reported & stat)) {
        g_reported |= stat;
        fprintf(stderr,
                "psprecomp: sceUmdWaitDriveStat(0x%02X) waits for a state this drive never"
                " reports (it reports 0x%02X) -- satisfied after one scheduling point,"
                " see src/hle/umd.c\n", stat, g_stat);
    }
    g_unsatisfied++;
    psp_threadman_yield();
    psp_ret(SCE_KERNEL_ERROR_OK);
}

static void hle_UmdCancelWaitDriveStat(void) {
    /* There is nothing to cancel: the wait above answers immediately rather
     * than parking a thread, so a thread can never be sitting on this drive.
     * Reporting success is then the truthful answer, not an approximation. */
    psp_ret(SCE_KERNEL_ERROR_OK);
}

static void hle_UmdRegisterUMDCallBack(void) {
    uint32_t cbid = psp_arg(0);

    /* An id that is not a callback is refused, as the firmware refuses it --
     * accepting it would leave a game believing it had a drive callback. The
     * check is against the toolkit's own callback table, so it is exactly as
     * wide as the callbacks the guest has actually created.
     *
     * Registering is not the end of it: a state change notifies this callback
     * (see notify_drive_state). A title that arms something from its drive
     * callback and waits on that flag is waiting for a state change now, which
     * is a real thing this model can produce, not for a missing runtime. */
    if (!psp_callback_valid(cbid)) { psp_ret(SCE_KERNEL_ERROR_ILLEGAL_ARGUMENT); return; }
    g_cbid = cbid;
    g_registrations++;
    psp_ret(SCE_KERNEL_ERROR_OK);
}

static void hle_UmdUnRegisterUMDCallBack(void) {
    uint32_t cbid = psp_arg(0);
    if (cbid != g_cbid) { psp_ret(SCE_KERNEL_ERROR_ILLEGAL_ARGUMENT); return; }
    g_cbid = 0;
    /* The firmware returns the id on older SDKs and zero on 6.x. A title being
     * brought up today is nearly always built against a recent SDK, and zero is
     * the answer that reads as success on both. */
    psp_ret(SCE_KERNEL_ERROR_OK);
}

void psp_umd_register(void) {
    /* NIDs are SHA-1(name)[0:4] little-endian, from pspsdk's sceUmdUser.S;
     * tests/test_hle.c verifies every pair registered here against the name.
     *
     * This is the whole of the library a title imports, and no more. The rest
     * of sceUmdUser -- CheckMedium, GetDiscInfo, GetErrorStat, the two
     * WaitDriveStat variants, ReplacePermit/Prohibit -- is left out until a
     * module imports it, because each of those would be a claim about the
     * medium that this file deliberately does not model. `allegrexrecomp funcs`
     * on a module says which ones it needs. */
    psp_hle_register(0xC6183D47, "sceUmdUser", "sceUmdActivate",            hle_UmdActivate);
    psp_hle_register(0xE83742BA, "sceUmdUser", "sceUmdDeactivate",          hle_UmdDeactivate);
    psp_hle_register(0x6B4A146C, "sceUmdUser", "sceUmdGetDriveStat",        hle_UmdGetDriveStat);
    psp_hle_register(0x8EF08FCE, "sceUmdUser", "sceUmdWaitDriveStat",       hle_UmdWaitDriveStat);
    psp_hle_register(0x6AF9B50A, "sceUmdUser", "sceUmdCancelWaitDriveStat", hle_UmdCancelWaitDriveStat);
    psp_hle_register(0xAEE7404D, "sceUmdUser", "sceUmdRegisterUMDCallBack", hle_UmdRegisterUMDCallBack);
    psp_hle_register(0xBD2BDE07, "sceUmdUser", "sceUmdUnRegisterUMDCallBack", hle_UmdUnRegisterUMDCallBack);
}
