/*
 * razor_abi_host_verify.c - ISO C11 / C++17 host conformance verifier for razor_abi_4byte_sys.h
 *
 * Verifies 64-byte cacheline layout, 16-byte Weft SU(2) rollback snapshot layout,
 * offset-0 4-byte host_ber_canary pointer compatibility, acquire-load polling,
 * and [16, 11, 4] Extended Hamming syndrome invariants.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../include/razor_abi_4byte_sys.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t popcnt16_host(uint32_t x) {
    uint32_t v = x & 0xFFFFu;
    uint32_t c = 0u;
    while (v != 0u) {
        v &= (v - 1u);
        c += 1u;
    }
    return c;
}

static uint32_t a16_secded_syndrome_host(uint32_t w16) {
    uint32_t p0 = popcnt16_host(w16 & 0xFFFFu) & 1u;
    uint32_t p1 = popcnt16_host(w16 & 0xAAAAu) & 1u;
    uint32_t p2 = popcnt16_host(w16 & 0xCCCCu) & 1u;
    uint32_t p3 = popcnt16_host(w16 & 0xF0F0u) & 1u;
    uint32_t p4 = popcnt16_host(w16 & 0xFF00u) & 1u;
    return p0 | (p1 << 1) | (p2 << 2) | (p3 << 3) | (p4 << 4);
}

int main(void) {
    HoloSphereRazorCanary64 canary;
    HoloSpherePiRollback16 rollback_q;
    memset(&canary, 0, sizeof(canary));
    rollback_q.w = 1.0f;
    rollback_q.x = 0.0f;
    rollback_q.y = 0.0f;
    rollback_q.z = 0.0f;

    if (((uintptr_t)&canary % 64u) != 0u) {
        fprintf(stderr, "FAIL: HoloSphereRazorCanary64 alignment %% 64 != 0\n");
        return 1;
    }
    if (((uintptr_t)&rollback_q % 16u) != 0u) {
        fprintf(stderr, "FAIL: HoloSpherePiRollback16 alignment %% 16 != 0\n");
        return 1;
    }

    uint32_t* legacy_canary_ptr = &canary.sec_count_u32;
    if ((const void*)legacy_canary_ptr != (const void*)&canary) {
        fprintf(stderr, "FAIL: offset-0 sec_count_u32 alias mismatch\n");
        return 1;
    }

    canary.sec_count_u32 = 16u;
    canary.ded_count_u32 = 1u;
    canary.epoch_u32 = 256u;
    canary.rollback_token_u32 = 192u;
    canary.non_lsb_canary_u32 = 2u;
    canary.last_syndrome_u32 = 0x1Fu;
    canary.ewma_ber_q16_u32 = 32u;
    canary.clean_streak_u32 = 16u;
    canary.hold_remaining_u32 = 4u;
    canary.ded_latch_remaining_u32 = 64u;

    HoloSphereRazorSnapshot snap = holosphere_razor_poll_acquire(&canary);
    if (snap.sec_count_u32 != 16u || snap.ded_count_u32 != 1u || snap.epoch_u32 != 256u ||
        snap.rollback_token_u32 != 192u || snap.non_lsb_canary_u32 != 2u ||
        snap.last_syndrome_u32 != 0x1Fu || snap.ewma_ber_q16_u32 != 32u ||
        snap.clean_streak_u32 != 16u || snap.hold_remaining_u32 != 4u ||
        snap.ded_latch_remaining_u32 != 64u) {
        fprintf(stderr, "FAIL: holosphere_razor_poll_acquire snapshot mismatch\n");
        return 1;
    }

    for (uint32_t i = 0u; i < 16u; ++i) {
        uint32_t s1 = a16_secded_syndrome_host(1u << i);
        if ((s1 & 1u) != 1u || ((s1 >> 1) & 0xFu) != i) {
            fprintf(stderr, "FAIL: 1-bit syndrome mismatch at lane %u\n", i);
            return 1;
        }
        for (uint32_t j = i + 1u; j < 16u; ++j) {
            uint32_t s2 = a16_secded_syndrome_host((1u << i) | (1u << j));
            if (s2 == 0u || (s2 & 1u) != 0u) {
                fprintf(stderr, "FAIL: 2-bit DED syndrome mismatch at (%u, %u)\n", i, j);
                return 1;
            }
        }
    }

    printf("{\"abi_version\":%u,\"canary_bytes\":%zu,\"rollback_bytes\":%zu,\"sec_lanes_verified\":16,\"ded_pairs_verified\":120,\"status\":\"OK\"}\n",
           HOLOSPHERE_RAZOR_ABI_VERSION,
           sizeof(HoloSphereRazorCanary64),
           sizeof(HoloSpherePiRollback16));
    return 0;
}
