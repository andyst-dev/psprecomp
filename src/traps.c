/* psprecomp — les deux appels que l'emetteur emet mais que rien n'implementait.
 *
 * Le generateur traduit la quasi-totalite d'un binaire, mais deux instructions
 * n'ont pas de traduction evidente :
 *
 *   - `syscall` : sur PSP c'est le chemin vers le noyau. On ne peut pas le
 *     traduire par un appel C, donc l'emetteur emet `psp_syscall(id)` et laisse
 *     le runtime decider.
 *   - tout ce qu'il ne traduit pas encore (`break`, les operations VFPU qu'il
 *     ne connait pas, les immediats pieges) : `psp_unimplemented(addr, nom)`.
 *
 * L'emetteur DECLARAIT ces deux fonctions dans l'en-tete genere, mais aucune
 * source du runtime ne les definissait : le C emis pour un jeu qui en contient
 * se compilait et ne se liait pas. Mesure sur LocoRoco (UCES00304) : 11 sites
 * `syscall` et 15 646 sites `psp_unimplemented`, et le lien echoue sur les deux
 * symboles tant que l'hote ne les fournit pas lui-meme.
 *
 * Le comportement est celui sur lequel repose le bring-up decrit dans
 * docs/BRINGUP.md : on compte, on parle un peu, on continue. C'est en regardant
 * ces messages disparaitre qu'on mesure l'avancement — "bringing a game up is
 * largely the process of watching those messages stop appearing".
 *
 * Le plafond d'affichage n'est pas cosmetique : un jeu qui boucle atteint des
 * millions d'appels par seconde et noierait tout le reste du journal. On affiche
 * les premiers, on compte les suivants, et les totaux restent lisibles a la fin.
 */
#include "psprecomp/recomp_rt.h"

#include <stdio.h>

#define PSP_TRAP_LOGGED 8

static uint64_t g_syscall_total;
static uint64_t g_unimplemented_total;
static uint32_t g_syscall_logged;
static uint32_t g_unimplemented_logged;

void psp_syscall(uint32_t id) {
    if (g_syscall_logged < PSP_TRAP_LOGGED) {
        g_syscall_logged++;
        fprintf(stderr, "psprecomp: syscall 0x%05X\n", id);
    }
    g_syscall_total++;
}

void psp_unimplemented(uint32_t addr, const char *what) {
    if (g_unimplemented_logged < PSP_TRAP_LOGGED) {
        g_unimplemented_logged++;
        fprintf(stderr, "psprecomp: unimplemented at 0x%08X: %s\n", addr,
                what ? what : "?");
    }
    g_unimplemented_total++;
}

uint64_t psp_syscall_total(void)       { return g_syscall_total; }
uint64_t psp_unimplemented_total(void) { return g_unimplemented_total; }

/* Un hote qui veut mesurer avant/apres une correction remet les compteurs a
 * zero, plutot que de comparer des totaux cumules depuis le demarrage. */
void psp_trap_reset(void) {
    g_syscall_total = 0;
    g_unimplemented_total = 0;
    g_syscall_logged = 0;
    g_unimplemented_logged = 0;
}
