#ifndef ENERGY_VAD_H
#define ENERGY_VAD_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WebRtcVadInst VadInst;

// 创建VAD实例
VadInst *energy_vad_create(void);
// 初始化VAD实例
int energy_vad_init(VadInst *vadInst);
// 设置VAD模式
int energy_vad_set_mode(VadInst *vadInst, int mode);
// 处理一帧音频
int energy_vad_process(VadInst *vadInst, uint32_t sampleRate, const int16_t *audio_frame, size_t frame_length, double energy_threshold);
// 释放VAD实例
void energy_vad_free(VadInst *vadInst);

#ifdef __cplusplus
}
#endif

#endif // ENERGY_VAD_H
