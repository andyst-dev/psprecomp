/*
 * scePower: the power callback slots.
 *
 * The firmware lets a game hand a callback over to be run when the power state
 * changes -- charger in or out, battery low, going to sleep. This HLE has no
 * power state: no battery, no charger, no suspend. So a callback registered
 * here is never fired, and that is a complete answer rather than a stub,
 * because nothing in the guest can make a power event happen -- the same shape
 * as the drive callback, which is why sceUmdUser registers one and this
 * registers one and neither ever runs.
 *
 * What is not optional is the slot. `scePowerRegisterCallback(-1, cbid)` asks
 * the kernel to pick a slot, and the *return value* is that slot number, not
 * zero: LocoRoco passes -1 and stores what comes back. So the auto-slot path
 * hands back a real index, and the argument is checked against the callback
 * table first -- a registration that records an id naming nothing is a lie that
 * surfaces much later, as a callback that never runs for no visible reason.
 */

#include "psprecomp/hle.h"

/* The firmware's slot count is not sourced here: 16 is the bound the SDK's
 * callback users assume, and the auto-slot path -- the only one LocoRoco uses,
 * three times, always with -1 -- never reaches it. A slot number is small on
 * hardware; any game that stores it only passes it back to unregister. */
#define MAX_POWER_SLOTS 16

static uint32_t g_slot[MAX_POWER_SLOTS];   /* callback id per slot, 0 = free */
static uint64_t g_registered;              /* registrations that succeeded */
static uint64_t g_conflicts;               /* refused: the id was not a callback, or the slot was taken */

void psp_power_reset(void) {
    for (int i = 0; i < MAX_POWER_SLOTS; i++) g_slot[i] = 0;
    g_registered = 0;
    g_conflicts = 0;
}

uint64_t psp_power_registered(void) { return g_registered; }

static void hle_PowerRegisterCallback(void) {
    /* (slot, cbid) */
    int32_t  slot = (int32_t)psp_arg(0);
    uint32_t cbid = psp_arg(1);

    if (!psp_callback_valid(cbid)) {
        g_conflicts++;
        psp_ret(SCE_KERNEL_ERROR_UNKNOWN_UID);
        return;
    }

    if (slot == -1) {
        for (int i = 0; i < MAX_POWER_SLOTS; i++) {
            if (g_slot[i]) continue;
            g_slot[i] = cbid;
            g_registered++;
            psp_ret((uint32_t)i);    /* the slot the kernel picked */
            return;
        }
        g_conflicts++;
        psp_ret(SCE_KERNEL_ERROR_NO_MEMORY);   /* no free slot */
        return;
    }

    if (slot < 0 || slot >= MAX_POWER_SLOTS) {
        g_conflicts++;
        psp_ret(SCE_KERNEL_ERROR_ILLEGAL_ARGUMENT);
        return;
    }
    if (g_slot[slot]) {
        /* Refused rather than overwritten: the first registration is the one
         * that would have been fired, so silently replacing it would drop the
         * game's own callback. (The exact code the kernel returns for a taken
         * slot is not sourced here; that it must not be accepted is.) */
        g_conflicts++;
        psp_ret(SCE_KERNEL_ERROR_ILLEGAL_ARGUMENT);
        return;
    }
    g_slot[slot] = cbid;
    g_registered++;
    psp_ret(SCE_KERNEL_ERROR_OK);
}

static void hle_PowerUnregisterCallback(void) {
    /* (slot) : the slot is the handle the game was given, so that is all it
     * passes back. An empty slot is accepted -- the game never calls this at
     * all, and the firmware's answer for an empty one is not sourced. */
    int32_t slot = (int32_t)psp_arg(0);
    if (slot < 0 || slot >= MAX_POWER_SLOTS) {
        psp_ret(SCE_KERNEL_ERROR_ILLEGAL_ARGUMENT);
        return;
    }
    g_slot[slot] = 0;
    psp_ret(SCE_KERNEL_ERROR_OK);
}

void psp_power_register(void) {
    psp_hle_register(0x04B7766E, "scePower", "scePowerRegisterCallback",   hle_PowerRegisterCallback);
    psp_hle_register(0xDFA8BAF8, "scePower", "scePowerUnregisterCallback", hle_PowerUnregisterCallback);
}

void psp_power_init(void) { psp_power_reset(); }
