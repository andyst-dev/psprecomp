/* Synthetic scheduler contract and lifecycle stress; no game data. */
#include "psprecomp/hle.h"
#include "psprecomp/dispatch.h"
#include "crypto/sha1.h"
#include <stdio.h>
#include <stdlib.h>

#define REQUIRE(c) do { if (!(c)) { \
    fprintf(stderr, "scheduler: line %d: %s\n", __LINE__, #c); exit(1); \
} } while (0)

static uint32_t call(const char *name, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    psp_cpu.r[PSP_REG_A0] = a;
    psp_cpu.r[PSP_REG_A1] = b;
    psp_cpu.r[PSP_REG_A2] = c;
    psp_cpu.r[PSP_REG_A3] = d;
    psp_hle_call(psp_nid(name));
    return psp_cpu.r[PSP_REG_V0];
}
static uint32_t create(uint32_t entry) {
    psp_cpu.r[PSP_REG_T0] = 0; /* fifth firmware argument: attributes */
    uint32_t id = call("sceKernelCreateThread", 0x08808000u, entry, 32, 0x4000);
    REQUIRE((id & 0xFFFF0000u) == 0x00040000u);
    return id;
}
static uint32_t sema, seen, worker_rc, reached, finished;
static void worker(void) {
    uint32_t token = psp_cpu.r[PSP_REG_A0];
    REQUIRE(psp_cpu.r[PSP_REG_A1] == 0x12345678u);
    REQUIRE((psp_cpu.r[PSP_REG_SP] & 15) == 0);
    /* A thread starts with its creator's $gp: a module's globals are
     * $gp-relative, and with 0 Pendemonium read its own code as data. */
    REQUIRE(psp_cpu.r[PSP_REG_GP] == 0x08A0C870u);
    psp_cpu.r[PSP_REG_S2] = token;
    reached++;
    worker_rc = call("sceKernelWaitSema", sema, 1, 0, 0);
    seen = psp_read32(0x08809000u);
    REQUIRE(psp_cpu.r[PSP_REG_S2] == token);
    finished++;
    psp_cpu.r[PSP_REG_V0] = token;
}
static void exits(void) {
    call("sceKernelExitThread", 0x55AAu, 0, 0, 0);
    REQUIRE(0); /* longjmp must bypass this */
}

int main(void) {
    REQUIRE(psp_mem_init() == 0);
    psp_cpu_reset();
    psp_cpu.r[PSP_REG_SP] = 0x08810000u;
    psp_cpu.r[PSP_REG_GP] = 0x08A0C870u;
    psp_hle_init();
    psp_register(0x08801000u, worker);
    psp_register(0x08801040u, exits);
    psp_write8(0x08808000u, 's');
    psp_write8(0x08808001u, 0);
    uint32_t free_before = psp_sysmem_free();
    /* More cycles than the thread table: slots, host handles and stacks must
     * be reusable. Four workers share an empty semaphore each round. */
    for (uint32_t round = 0; round < 80; round++) {
        reached = finished = seen = 0;
        psp_write32(0x08809000u, 0);
        psp_cpu.r[PSP_REG_T0] = 0;
        sema = call("sceKernelCreateSema", 0x08808000u, 0, 0, 4);
        uint32_t ids[4];
        for (uint32_t i = 0; i < 4; i++) {
            ids[i] = create(0x08801000u);
            psp_cpu.r[PSP_REG_S2] = 0xABCD0000u + i;
            REQUIRE(call("sceKernelStartThread", ids[i], i + 1, 0x12345678u, 0) == 0);
            REQUIRE(psp_cpu.r[PSP_REG_S2] == 0xABCD0000u + i);
        }
#if PSPRECOMP_REAL_THREADS
        REQUIRE(reached == 0); /* StartThread must return before any body */
        psp_threadman_yield();
        REQUIRE(reached == 4); /* pending first acquisitions get the baton */
        REQUIRE(finished == 0); /* no false success on the empty semaphore */
#else
        REQUIRE(reached == 4 && finished == 4); /* inline, not queued */
        REQUIRE(worker_rc == SCE_KERNEL_ERROR_WAIT_TIMEOUT);
        REQUIRE(seen == 0);
#endif
        psp_write32(0x08809000u, 0xC0FFEEu); /* work posted AFTER StartThread */
        REQUIRE(call("sceKernelSignalSema", sema, 4, 0, 0) == 0);
        for (uint32_t i = 0; i < 4; i++) {
            REQUIRE(call("sceKernelWaitThreadEnd", ids[i], 0x08809004u, 0, 0) == 0);
            REQUIRE(psp_read32(0x08809004u) == i + 1);
            REQUIRE(call("sceKernelDeleteThread", ids[i], 0, 0, 0) == 0);
        }
#if PSPRECOMP_REAL_THREADS
        REQUIRE(worker_rc == 0 && seen == 0xC0FFEEu && finished == 4);
#endif
        REQUIRE(call("sceKernelDeleteSema", sema, 0, 0, 0) == 0);
        REQUIRE(psp_sysmem_free() == free_before);
    }
    /* Restart an ended thread, including the explicit ExitThread unwind. */
    uint32_t id = create(0x08801040u);
    for (int i = 0; i < 8; i++) {
        REQUIRE(call("sceKernelStartThread", id, 0, 0, 0) == 0);
        REQUIRE(call("sceKernelWaitThreadEnd", id, 0x08809004u, 0, 0) == 0);
        REQUIRE(psp_read32(0x08809004u) == 0x55AAu);
    }
    REQUIRE(call("sceKernelDeleteThread", id, 0, 0, 0) == 0);
    REQUIRE(psp_sysmem_free() == free_before);
    psp_threadman_reset(); /* joins ended hosts and destroys their locks */
    REQUIRE(psp_threadman_created() == 0 && psp_threadman_started() == 0);
    psp_mem_free();
    printf("scheduler (%s): 320 worker lifecycles + 8 restarts passed\n",
           PSPRECOMP_REAL_THREADS ? "host threads" : "inline default");
    return 0;
}
