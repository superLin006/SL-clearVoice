#ifndef RK_AUDIOEFFECT_INTERFACE_H
#define RK_AUDIOEFFECT_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

// EQ_DRC相关接口
extern void* AudioPost_Init(float* pfPara, int swFrmLen);

extern void* AudioPost_InitByPath(char* pfParaPath, int swFrmLen);

extern int AudioPost_Process(void* st_ptr_, short int* pfIn, short int* pfOut, short int shwChannelNum, int swFrmLen);

extern int AudioPost_Destroy(void* st_ptr_);

extern int AudioPost_SetPara(void* st_ptr_, float *pfPara,int swFrmLen);

extern int AudioPost_GetPara(void* st_ptr_, float *pfPara);

extern int AudioPost_EQ10_SetPara(void* st_ptr_, float *pstEQ10);

extern int AudioPost_EQ10_GetPara(void* st_ptr_, float *pstEQ10);

extern int AudioPost_EQ10_GetData(void* st_ptr_, float** pshwIn, float** pshwOut);

extern int AudioPost_MultiBandDRC_Crossover_AdjustPara(void* st_ptr_, int i_ch, float fc_low, float fc_high, float fc_med1_hp, float fc_med1_lp, float fc_med2_hp, float fc_med2_lp);

extern int AudioPost_MultiBandDRC_DRC_AdjustPara(void* st_ptr_, int i_ch, int i_band, float gain_dB, int drc_enable, float compress_start, float expand_end, float noise_threshold, float max_gain, float max_peak, float attack_time, float release_time, float hold_time);

extern int AudioPost_Agc_AdjustPara(void* st_ptr_, int i_ch, float compress_start, float expand_end, float noise_threshold, float max_gain, float max_peak, float attack_time, float release_time, float hold_time);

extern int AudioPost_Maximizer_AdjustPara(void* st_ptr_, int i_ch, float max_threshold, float ceiling, float release);

#ifdef __cplusplus
} 
#endif

#endif 