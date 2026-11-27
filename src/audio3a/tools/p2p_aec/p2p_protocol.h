#ifndef P2P_PROTOCOL_H
#define P2P_PROTOCOL_H

#include <stdint.h>

#define P2P_MAGIC           0x41454331  /* "AEC1" */
#define P2P_DEFAULT_DUT_IP  "172.16.40.172"
#define P2P_DEFAULT_PEER_IP "172.16.40.208"
#define P2P_PORT_TO_DUT     9998        /* Peer -> DUT (Far-end audio) */
#define P2P_PORT_FROM_DUT   9999        /* DUT -> Peer (Clean/Processed audio) */

#define P2P_SAMPLE_RATE     16000
#define P2P_FRAME_SAMPLES   256         /* 16 ms per frame */
#define P2P_PAYLOAD_BYTES   (P2P_FRAME_SAMPLES * sizeof(int16_t)) /* 512 bytes */

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;         /* P2P_MAGIC */
    uint32_t seq;           /* Frame sequence number */
    uint32_t timestamp_ms;  /* Sender timestamp in ms */
    uint16_t samples;       /* Number of samples (256) */
    uint16_t flags;         /* Bit 0: 0=Raw, 1=AEC processed */
    int16_t  pcm[P2P_FRAME_SAMPLES]; /* 16-bit PCM audio */
} P2PPacket;
#pragma pack(pop)

#endif /* P2P_PROTOCOL_H */

