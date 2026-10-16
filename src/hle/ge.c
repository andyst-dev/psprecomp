/* psprecomp — sceGe_user.
 *
 * The GE is the PSP's GPU. It is not driven by function calls: user code builds
 * a **display list** — an array of 32-bit words, each an 8-bit command and 24
 * bits of argument — and hands the GE a pointer plus a *stall address*. The GE
 * consumes commands up to the stall, and the CPU moves the stall forward as it
 * writes more. That producer/consumer arrangement is the whole API.
 *
 * So `sceGu*` (the list-building library) is ordinary user code and gets
 * recompiled like anything else. Only list *execution* is emulated, and that is
 * this file.
 *
 * ## What this does and does not do
 *
 * It walks the list, follows control flow (JUMP/CALL/RET/END/FINISH), and
 * tracks the state commands that matter — framebuffer, vertex format,
 * primitive counts. It does **not rasterize**. No triangles are drawn.
 *
 * That is deliberately the useful half to build first. During bring-up the
 * question is not "does it look right" but "is the game drawing anything at
 * all, and what?" — and a command-stream summary answers that, while a
 * half-working rasterizer answers it misleadingly. psp_ge_dump_stats() reports
 * what the game asked for; making those triangles appear is a separate phase
 * with its own correctness problem.
 */

#include "psprecomp/hle.h"
#include "psprecomp/render.h"

#include <stdio.h>
#include <string.h>

/* Display-list opcodes. Only the ones the walk needs to be correct about are
 * named; everything else is counted rather than guessed at, because a
 * misidentified state command silently changes rendering. */
#define GE_NOP          0x00
#define GE_VADDR        0x01
#define GE_IADDR        0x02
#define GE_PRIM         0x04
#define GE_BEZIER       0x05
#define GE_SPLINE       0x06
#define GE_JUMP         0x08
#define GE_BJUMP        0x09
#define GE_CALL         0x0A
#define GE_RET          0x0B
#define GE_END          0x0C
#define GE_SIGNAL       0x0E
#define GE_FINISH       0x0F
#define GE_BASE         0x10
#define GE_VTYPE        0x12
#define GE_OFFSET_ADDR  0x13
#define GE_ORIGIN_ADDR  0x14
#define GE_FBP          0x9C
#define GE_FBW          0x9D

#define MAX_QUEUES 8
#define GE_STACK   8

/* Primitive types, from the PRIM argument's type field. */
static const char *const PRIM_NAME[8] = {
    "points", "lines", "line-strip", "triangles",
    "triangle-strip", "triangle-fan", "sprites", "?"
};

typedef struct {
    uint32_t id;
    uint32_t list;      /* current read pointer */
    uint32_t stall;     /* stop before this address; 0 means "no stall" */
    uint32_t base;      /* GE_BASE: high bits for addresses */
    uint32_t origin;
    int      used;
    int      done;
} ge_queue;

static ge_queue g_queue[MAX_QUEUES];
static uint32_t g_next_id;

static uint64_t g_unsupported;     /* vertices in a format we do not read */

/* Which state commands the guest actually programs, and with which argument.
 *
 * The interpreter decodes the flow and the few commands it needs in order to
 * draw; every other command lands in one bucket. During bring-up the question is
 * not how many were skipped but WHICH ones, so they are counted per opcode with
 * the first and last argument seen. They are named only where the toolkit
 * already knows the name -- an invented name reads like a fact, whereas a number
 * with its argument is a measurement that can be checked against the guest. The
 * first run said the guest opens with a zeroing sweep of 0x3B..0x5D, which is
 * the matrix and viewport block; what they carry later is the question. */
static uint64_t g_cmd_count[256];
static uint32_t g_cmd_first[256];
static uint32_t g_cmd_last[256];

/* The colours the guest actually submits, for the vertex formats that carry one.
 * "The frame is white" and "the guest sends white" look the same until these are
 * counted. */
#define COL_HIST 8
static uint32_t g_col_val[COL_HIST];
static uint64_t g_col_count[COL_HIST];
static uint64_t g_col_total;
static unsigned g_col_n;

/* Vertices drawn per vertex format. Read against pspsdk's GU_* shifts
 * (GU_VERTEX_SHIFT(n) = n << 7, GU_COLOR_SHIFT(n) = n << 2, GU_TRANSFORM_2D =
 * 1 << 23), so a format this toolkit cannot read yet is a count rather than a
 * guess about what the game is doing. */
#define VT_HIST 16
static uint32_t g_vt_val[VT_HIST];
static uint64_t g_vt_verts[VT_HIST];
static unsigned g_vt_n;

/* Tracked state, and the counters that make the report worth reading. */
static struct {
    uint32_t fbp, fbw, vtype, vaddr;
    uint64_t commands;
    uint64_t prims[8];
    uint64_t vertices;
    uint64_t unknown;
    uint64_t lists;
    uint64_t finishes;
} g_ge;

void psp_ge_reset(void) {
    memset(g_queue, 0, sizeof g_queue);
    memset(&g_ge, 0, sizeof g_ge);
    memset(g_cmd_count, 0, sizeof g_cmd_count);
    memset(g_cmd_first, 0, sizeof g_cmd_first);
    memset(g_cmd_last, 0, sizeof g_cmd_last);
    memset(g_vt_val, 0, sizeof g_vt_val);
    memset(g_vt_verts, 0, sizeof g_vt_verts);
    memset(g_col_val, 0, sizeof g_col_val);
    memset(g_col_count, 0, sizeof g_col_count);
    g_col_total = 0;
    g_col_n = 0;
    g_vt_n = 0;
    psp_render_reset_pixels();
    g_unsupported = 0;
    g_next_id = 0x00080000u;
}

void psp_ge_init(void) { psp_ge_reset(); }

/* Hand the backend the framebuffer being drawn into, in *guest* addresses.
 *
 * The GE's own frame addresses are offsets inside eDRAM (FBP 0 is the first byte
 * of VRAM), so they become guest addresses by adding the VRAM window the guest
 * maps -- the same base its sceGeEdramGetAddr answers and the one it passes to
 * sceDisplaySetFrameBuf. Passing the raw offset is not a cosmetic error:
 * measured on LocoRoco, the game drew with FBP 0 while its display buffer was
 * 0x04000000, and 138 774 pixels were written at 0x00000000 -- over the module
 * image -- with a black screen. */
static void set_ge_target(void) {
    uint32_t rel = g_ge.fbp & 0x00FFFFFFu;
    psp_render_current()->set_target(PSP_VRAM_BASE + rel, g_ge.fbw, 0);
}

uint64_t psp_ge_state_command_count(unsigned cmd, uint32_t *first, uint32_t *last) {
    if (cmd > 255) return 0;
    if (first) *first = g_cmd_first[cmd];
    if (last)  *last  = g_cmd_last[cmd];
    return g_cmd_count[cmd];
}

void psp_ge_dump_stats(FILE *out) {
    fprintf(out, "GE: %llu lists, %llu commands, %llu finishes\n",
            (unsigned long long)g_ge.lists,
            (unsigned long long)g_ge.commands,
            (unsigned long long)g_ge.finishes);
    fprintf(out, "    framebuffer 0x%08X stride %u, vertex type 0x%06X\n",
            g_ge.fbp, g_ge.fbw, g_ge.vtype);
    fprintf(out, "    vertices submitted: %llu\n", (unsigned long long)g_ge.vertices);
    for (int i = 0; i < 8; i++)
        if (g_ge.prims[i])
            fprintf(out, "    %-15s %llu\n", PRIM_NAME[i], (unsigned long long)g_ge.prims[i]);
    if (g_ge.unknown) {
        fprintf(out, "    %llu commands not individually decoded, by opcode "
                     "(1er et dernier argument) :\n",
                (unsigned long long)g_ge.unknown);
        /* TOUS les opcodes vus, dans l'ordre des numeros : un top 10 cache
         * precisement ceux qui apparaissent une fois par image -- un format, une
         * taille, une adresse -- c'est-a-dire l'etat qui configure le dessin. */
        for (unsigned c = 0; c < 256; c++)
            if (g_cmd_count[c])
                fprintf(out, "        0x%02X %10llu fois, 1er arg 0x%06X, dernier 0x%06X\n",
                        c, (unsigned long long)g_cmd_count[c],
                        g_cmd_first[c], g_cmd_last[c]);
    }
    if (g_vt_n) {
        fprintf(out, "    vertices per vertex format (GU_* shifts: vertex 7, color 2, "
                     "transform-2D bit 23):\n");
        for (unsigned i = 0; i < g_vt_n; i++)
            fprintf(out, "        0x%06X %10llu%s\n", g_vt_val[i],
                    (unsigned long long)g_vt_verts[i],
                    ((g_vt_val[i] >> 23) & 1) ? "" : "   (not pass-through: not drawn)");
    }
    if (g_col_total) {
        fprintf(out, "    vertex colours submitted (8888 only; the other formats "
                     "are not decoded and draw white), %llu au total :\n",
                (unsigned long long)g_col_total);
        for (unsigned i = 0; i < g_col_n; i++)
            fprintf(out, "        0x%08X %10llu\n", g_col_val[i],
                    (unsigned long long)g_col_count[i]);
    }
    fprintf(out, "    pixels written: %llu\n", (unsigned long long)psp_render_pixels());
    if (g_unsupported)
        fprintf(out, "    %llu vertices in a format this interpreter does not process "
                     "(not pass-through, no position, or no vertex list)\n",
                (unsigned long long)g_unsupported);
}

uint64_t psp_ge_command_count(void) { return g_ge.commands; }
uint64_t psp_ge_vertex_count(void)  { return g_ge.vertices; }

/* ---- rasterizer ----------------------------------------------------------
 *
 * Enough of the GE to put pixels in the framebuffer. Deliberately narrow:
 *
 *  - "through" mode only (VTYPE bit 23), where vertex coordinates are already
 *    in screen space. That is what 2D games and every UI layer use. Transformed
 *    geometry needs the matrix pipeline and is not attempted here -- drawing it
 *    with the wrong transform would look like a rendering bug rather than a
 *    missing feature.
 *  - Position as 16-bit or float; colour as 8888 or none.
 *  - Flat/interpolated colour, no texturing, no depth, no blending.
 *
 * The point is to close the loop from display list to visible pixels so the
 * rest can be measured against something. Everything omitted is omitted
 * loudly: unsupported vertex formats are counted, not guessed at.
 */

/* VTYPE field extraction. */
#define VT_TEX(v)     ((v) & 3)
#define VT_COLOR(v)   (((v) >> 2) & 7)
#define VT_NORMAL(v)  (((v) >> 5) & 3)
#define VT_POS(v)     (((v) >> 7) & 3)
#define VT_WEIGHT(v)  (((v) >> 9) & 3)
#define VT_INDEX(v)   (((v) >> 11) & 3)
#define VT_THROUGH(v) (((v) >> 23) & 1)


uint64_t psp_ge_pixels(void) { return psp_render_pixels(); }

/* Size of one vertex in bytes, and the offsets within it. Components appear in
 * a fixed order (weights, texture, colour, normal, position) and each is
 * aligned to its own size, which is what makes the stride awkward enough to be
 * worth computing rather than assuming. */
static int vertex_layout(uint32_t vtype, int *col_off, int *pos_off) {
    static const int tex_sz[4]   = { 0, 1, 2, 4 };
    static const int col_sz[8]   = { 0, 0, 0, 0, 2, 2, 2, 4 };
    static const int norm_sz[4]  = { 0, 1, 2, 4 };
    static const int pos_sz[4]   = { 0, 1, 2, 4 };

    int off = 0, align = 1;
    int t = tex_sz[VT_TEX(vtype)] * 2;
    int c = col_sz[VT_COLOR(vtype)];
    int n = norm_sz[VT_NORMAL(vtype)] * 3;
    int p = pos_sz[VT_POS(vtype)] * 3;

    if (VT_WEIGHT(vtype)) return 0;          /* skinning: not handled */

    int ts = tex_sz[VT_TEX(vtype)];
    if (ts) { off = (off + ts - 1) & ~(ts - 1); off += t; if (ts > align) align = ts; }
    int cs = col_sz[VT_COLOR(vtype)];
    if (cs) { off = (off + cs - 1) & ~(cs - 1); *col_off = off; off += c; if (cs > align) align = cs; }
    else *col_off = -1;
    int ns = norm_sz[VT_NORMAL(vtype)];
    if (ns) { off = (off + ns - 1) & ~(ns - 1); off += n; if (ns > align) align = ns; }
    int ps = pos_sz[VT_POS(vtype)];
    if (!ps) return 0;                        /* no position: nothing to draw */
    off = (off + ps - 1) & ~(ps - 1); *pos_off = off; off += p;
    if (ps > align) align = ps;

    return (off + align - 1) & ~(align - 1);  /* stride */
}

/* Les couleurs reellement soumises par l'invite, pour les formats qui en portent
 * une. « L'image est blanche » et « l'invite envoie du blanc » se ressemblent
 * exactement jusqu'a ce qu'on les compte. Table pleine : on remplace la couleur
 * la moins vue, donc le rapport garde les plus frequentes et non les premieres
 * rencontrees. */
static void note_colour(uint32_t c) {
    g_col_total++;
    for (unsigned i = 0; i < g_col_n; i++)
        if (g_col_val[i] == c) { g_col_count[i]++; return; }
    if (g_col_n < COL_HIST) { g_col_val[g_col_n] = c; g_col_count[g_col_n] = 1; g_col_n++; return; }
    unsigned least = 0;
    for (unsigned i = 1; i < COL_HIST; i++)
        if (g_col_count[i] < g_col_count[least]) least = i;
    g_col_val[least] = c;
    g_col_count[least] = 1;
}

static int read_vertex(uint32_t addr, uint32_t vtype, int col_off, int pos_off,
                       psp_vertex *out) {
    out->rgba = 0xFFFFFFFFu;
    if (col_off >= 0 && VT_COLOR(vtype) == 7) {
        out->rgba = psp_read32(addr + (uint32_t)col_off);
        note_colour(out->rgba);
    }

    switch (VT_POS(vtype)) {
    case 2:   /* 16-bit */
        out->x = (int16_t)psp_read16(addr + (uint32_t)pos_off);
        out->y = (int16_t)psp_read16(addr + (uint32_t)pos_off + 2);
        return 1;
    case 3: { /* float */
        out->x = (int)psp_read_f32(addr + (uint32_t)pos_off);
        out->y = (int)psp_read_f32(addr + (uint32_t)pos_off + 4);
        return 1;
    }
    default:
        return 0;
    }
}


static void draw_prim(uint32_t type, uint32_t count) {
    for (unsigned i = 0; i < g_vt_n; i++)
        if (g_vt_val[i] == g_ge.vtype) { g_vt_verts[i] += count; goto noted; }
    if (g_vt_n < VT_HIST) { g_vt_val[g_vt_n] = g_ge.vtype; g_vt_verts[g_vt_n] = count; g_vt_n++; }
noted:
    if (!VT_THROUGH(g_ge.vtype)) { g_unsupported += count; return; }
    if (!g_ge.vaddr) { g_unsupported += count; return; }

    int col_off = -1, pos_off = 0;
    int stride = vertex_layout(g_ge.vtype, &col_off, &pos_off);
    if (!stride) { g_unsupported += count; return; }

    /* Decode the whole batch, then hand it to the backend in one call.
     *
     * Format decoding stays here rather than in each backend: the stride
     * arithmetic and component alignment are fiddly, and duplicating them per
     * backend means every backend is wrong in its own way. Wrong once,
     * centrally, is at least diagnosable. */
    enum { BATCH = 256 };
    psp_vertex v[BATCH];
    const psp_render_backend *be = psp_render_current();

    uint32_t done = 0;
    while (done < count) {
        uint32_t n = count - done;
        if (n > BATCH) n = BATCH;

        /* Strips are order-dependent, so a batch boundary must overlap by two
         * vertices or the triangle spanning it is lost. */
        uint32_t decoded = 0;
        for (; decoded < n; decoded++) {
            if (!read_vertex(g_ge.vaddr + (done + decoded) * (uint32_t)stride,
                             g_ge.vtype, col_off, pos_off, &v[decoded]))
                break;
        }
        if (!decoded) break;

        be->draw((int)type, v, (int)decoded);

        if (type == 4 && decoded == BATCH && done + decoded < count)
            done += decoded - 2;     /* strip overlap */
        else
            done += decoded;
    }
}

/* Walk a list until END/FINISH, the stall address, or a step budget.
 *
 * The budget is not paranoia: a list whose JUMP forms a cycle is a normal
 * intermediate state while the CPU is still writing, and without a bound a
 * malformed or partially-written list hangs the host with no diagnostic. */
static void run_list(ge_queue *q) {
    uint32_t stack[GE_STACK];
    int sp = 0;
    uint64_t budget = 1u << 22;

    g_ge.lists++;

    while (budget--) {
        if (q->stall && q->list == q->stall) break;   /* caught up to the CPU */

        uint32_t word = psp_read32(q->list);
        uint32_t cmd  = word >> 24;
        uint32_t arg  = word & 0x00FFFFFF;
        q->list += 4;
        g_ge.commands++;

        switch (cmd) {
        case GE_NOP:
            break;

        case GE_PRIM: {
            uint32_t type  = (arg >> 16) & 7;
            uint32_t count = arg & 0xFFFF;
            g_ge.prims[type]++;
            g_ge.vertices += count;
            draw_prim(type, count);
            break;
        }
        case GE_BEZIER:
        case GE_SPLINE:
            /* Patches expand to triangles on hardware; counted as their own
             * thing rather than folded into the triangle count. */
            g_ge.prims[3]++;
            break;

        case GE_JUMP:
            q->list = (q->base | (arg & 0xFFFFFC));
            break;
        case GE_CALL:
            if (sp < GE_STACK) stack[sp++] = q->list;
            q->list = (q->base | (arg & 0xFFFFFC));
            break;
        case GE_RET:
            if (sp > 0) q->list = stack[--sp];
            break;
        case GE_BJUMP:
            /* Conditional on the bounding-box test, which needs geometry we do
             * not process. Not taking it means we walk the enclosed commands
             * rather than skipping them -- the conservative direction, since
             * skipping would under-report what the game drew. */
            break;

        case GE_END:
        case GE_FINISH:
            if (cmd == GE_FINISH) g_ge.finishes++;
            q->done = 1;
            return;

        case GE_SIGNAL:
            /* Raises a callback on hardware. Callbacks are not delivered yet
             * (no scheduler), so this is recorded and ignored. */
            break;

        case GE_BASE:        q->base = (arg & 0xFF0000) << 8; break;
        case GE_ORIGIN_ADDR: q->origin = q->list - 4; break;
        case GE_OFFSET_ADDR: q->base = arg << 8; break;

        case GE_VTYPE: g_ge.vtype = arg; break;
        /* The GE's frame addresses are *offsets inside eDRAM*: FBP 0 is the first
         * byte of VRAM, which the guest maps at PSP_VRAM_BASE -- the address its
         * own sceGeEdramGetAddr answers and the one it passes to
         * sceDisplaySetFrameBuf. Geometry drawn with the raw offset lands at
         * 0x00000000, i.e. over the module image, while the screen stays black.
         * Measured before this: the game set FBP 0 with its display buffer at
         * 0x04000000, and 138 774 pixels went into the bottom of the module. */
        case GE_FBP:
            g_ge.fbp = (g_ge.fbp & 0xFF000000u) | arg;
            set_ge_target();
            break;
        case GE_FBW:
            g_ge.fbw = arg & 0xFFFF;
            g_ge.fbp = (g_ge.fbp & 0x00FFFFFFu) | ((arg & 0xFF0000) << 8);
            set_ge_target();
            break;

        case GE_VADDR: g_ge.vaddr = (q->base | (arg & 0xFFFFFF)); break;
        case GE_IADDR: break;

        default:
            /* Un vrai changement d'etat que l'interpreteur ne decode pas
             * individuellement. Compte par opcode avec son premier et son dernier
             * argument : « N commandes non decodees » cache si l'invite configure
             * un dessin ou pousse de l'etat, ce qui est la difference entre un
             * defaut de rendu et un jeu qui n'a pas encore demande a rendre. */
            if (g_cmd_count[cmd]++ == 0) g_cmd_first[cmd] = arg;
            g_cmd_last[cmd] = arg;
            if (g_ge.unknown < 64)
                fprintf(stderr, "  GE cmd 0x%02X arg 0x%06X\n", cmd, arg);
            g_ge.unknown++;
            break;
        }
    }
}

/* ---- the calls ----------------------------------------------------------- */

static ge_queue *find_queue(uint32_t id) {
    for (int i = 0; i < MAX_QUEUES; i++)
        if (g_queue[i].used && g_queue[i].id == id) return &g_queue[i];
    return NULL;
}

static void enqueue(int head) {
    /* (list, stall, cbid, arg) */
    ge_queue *q = NULL;
    for (int i = 0; i < MAX_QUEUES; i++) if (!g_queue[i].used) { q = &g_queue[i]; break; }
    if (!q) { psp_ret(SCE_KERNEL_ERROR_NO_MEMORY); return; }

    memset(q, 0, sizeof *q);
    q->id    = g_next_id++;
    q->list  = psp_arg(0) & ~3u;
    q->stall = psp_arg(1) & ~3u;
    q->used  = 1;
    (void)head;

    /* Hardware runs the list asynchronously. We run it here and finish before
     * returning, which is indistinguishable from the game's point of view
     * because every way it can observe progress -- ListSync, DrawSync -- then
     * reports completion.
     *
     * Which is also why the slot is released here: holding it would cap a title
     * at MAX_QUEUES lists *for the whole run*, and every later enqueue would
     * fail. Measured on LocoRoco: the run reported exactly 8 lists -- the size of
     * this table -- and the 9th enqueue returned an error the game cannot see,
     * so it would simply stop drawing. A game re-enqueues once per frame. */
    run_list(q);
    uint32_t id = q->id;
    q->used = 0;
    psp_ret(id);
}

static void hle_ListEnQueue(void)     { enqueue(0); }
static void hle_ListEnQueueHead(void) { enqueue(1); }

static void hle_ListUpdateStallAddr(void) {
    ge_queue *q = find_queue(psp_arg(0));
    if (!q) { psp_ret(SCE_KERNEL_ERROR_UNKNOWN_UID); return; }
    q->stall = psp_arg(1) & ~3u;
    if (!q->done) run_list(q);          /* the new stall released more commands */
    psp_ret(SCE_KERNEL_ERROR_OK);
}

/* Lists are complete by the time they are enqueued, so every sync succeeds
 * immediately. */
static void hle_ListSync(void) { psp_ret(SCE_KERNEL_ERROR_OK); }
static void hle_DrawSync(void) { psp_ret(SCE_KERNEL_ERROR_OK); }

static void hle_Break(void)    { psp_ret(SCE_KERNEL_ERROR_OK); }
static void hle_Continue(void) { psp_ret(SCE_KERNEL_ERROR_OK); }

static void hle_SetCallback(void)   { psp_ret(0); }
static void hle_UnsetCallback(void) { psp_ret(SCE_KERNEL_ERROR_OK); }

/* eDRAM is the GPU-visible VRAM window: 2 MB at 0x04000000. */
static void hle_EdramGetAddr(void) { psp_ret(PSP_VRAM_BASE); }
static void hle_EdramGetSize(void) { psp_ret(PSP_VRAM_SIZE); }

void psp_ge_register(void) {
    psp_hle_register(0xAB49E76A, "sceGe_user", "sceGeListEnQueue",         hle_ListEnQueue);
    psp_hle_register(0x1C0D95A6, "sceGe_user", "sceGeListEnQueueHead",     hle_ListEnQueueHead);
    psp_hle_register(0xE0D68148, "sceGe_user", "sceGeListUpdateStallAddr", hle_ListUpdateStallAddr);
    psp_hle_register(0x03444EB4, "sceGe_user", "sceGeListSync",            hle_ListSync);
    psp_hle_register(0xB287BD61, "sceGe_user", "sceGeDrawSync",            hle_DrawSync);
    psp_hle_register(0xB448EC0D, "sceGe_user", "sceGeBreak",               hle_Break);
    psp_hle_register(0x4C06E472, "sceGe_user", "sceGeContinue",            hle_Continue);
    psp_hle_register(0xA4FC06A4, "sceGe_user", "sceGeSetCallback",         hle_SetCallback);
    psp_hle_register(0x05DB22CE, "sceGe_user", "sceGeUnsetCallback",       hle_UnsetCallback);
    psp_hle_register(0xE47E40E4, "sceGe_user", "sceGeEdramGetAddr",        hle_EdramGetAddr);
    psp_hle_register(0x1F6752AD, "sceGe_user", "sceGeEdramGetSize",        hle_EdramGetSize);
}
