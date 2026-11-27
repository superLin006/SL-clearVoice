#ifndef _RKAUDIO_PREPROCESS_H_
#define _RKAUDIO_PREPROCESS_H_

#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif
//#define ENABLE_STUDIO 1

#define NUM_CHANNEL					 16
#define NUM_REF_CHANNEL              2
#define NUM_DROP_CHANNEL             0
#define REF_POSITION				 1

	static float Array[NUM_CHANNEL] = { 2, 8, 9, 0, 1, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15 };

	/**********************EQ Parameter(Deprecated)********************/
	static float EqPara_16k[5][13] =
	{
	//filter_bank 1
	{-1 ,-2 ,-1 ,0 ,0 ,-3 ,-4 ,-1 ,0 ,-3 ,-6 ,-4 ,-1 },
	//filter_bank 2
	{-1 ,-4 ,-5 ,-6 ,-10 ,-9 ,-3 ,-3 ,-11 ,-12 ,2 ,12 ,0 },
	//filter_bank 3
	{-16 ,-11 ,2 ,-1 ,-11 ,-8 ,-4 ,-10 ,-6 ,13 ,7 ,-40 ,-62 },
	//filter_bank 4
	{-17 ,23 ,-22 ,-83 ,-57 ,6 ,-14 ,-92 ,-126 ,-141 ,-200 ,-197 ,-54 },
	//filter_bank 5
	{-8 ,-249 ,-390 ,10 ,428 ,-142 ,-1341 ,-1365 ,208 ,664 ,-2836 ,-8715 ,32764 },
	};

	/**********************AIEC Parameter**********************/
	static float Ratio[3] = {1.0f, 1.0f, 1.0f};  //ratio
		
	/**********************AES Parameter**********************/
	static float LimitRatio[2][3] = {
		/* low freq   median freq    high freq*/
		{   2.5f,        1.2f,          1.0f  },  //limit
		{   1.0f,        0.8f,          0.6f  },  //ratio
	};
	/**********************THD Parameter**********************/
	static float ThdSplitFreq[4][2] = {
		{ 400,3800},//For low frequency
		{ 3800,8000},
		{ 0,0},//{ 3500,8000},
		{ 0,0},
	};

	static float ThdSupDegree[4][10] =
	{
		/* 2th		3th		4th		5th		6th		7th		8th		9th		10th	11th order  */
		{0.0005f, 0.0005f,0.0005f,	0.0005f,	0,	0,		0,		0,		0,		0},
		{ 0.0003f, 0.0003f,   0.0003f, 0.0003f,	0,	0,	0,		0,		0,		0},
		{ 0,		0,		0,		0,	0.0002f, 0.0002f, 0.0002f,0,	0,		0,},
		{ 0.003f, 0.003f, 0.004f,	0.005f, 0.003f, 0.003f, 0.003f,	0,		0,		0},
	};
	static float HardSplitFreq[5][2] = {
		{ 400,1500},	//1 to 4 is select hard suppress freq bin
		{ 0,0},
		{ 0,0},
		{ 0,0},
		{ 1500,5000},//freq use to calculate mean_G
	};
	static float HardThreshold[4] = { 0.15,0.15, 0.25, 0.15 };

	/*************************************************/
	/*The Main Enable which used to control the AEC,BF and RX*/
	typedef enum RKAUDIOEnable_
	{
		RKAUDIO_EN_AEC = 1 << 0,
		RKAUDIO_EN_BF = 1 << 1,
		RKAUDIO_EN_RX = 1 << 2,
		RKAUDIO_EN_CMD = 1 << 3,
	} RKAUDIOEnable;

	/* The Sub-Enable which used to control the AEC,BF and RX*/
	typedef enum RKAecEnable_
	{
		EN_DELAY = 1 << 0,
		EN_ARRAY_RESET = 1 << 1,
	} RKAecEnable;
	
	typedef enum RKPreprocessEnable_
	{
		EN_Fastaec = 1 << 0,
		EN_Wakeup = 1 << 1,
		EN_Dereverberation = 1 << 2,
		EN_AES = 1 << 3,
		EN_Agc = 1 << 4,
		EN_Anr = 1 << 5,
		EN_STDT = 1 << 7,
		EN_CNG = 1 << 8,
		EN_EQ = 1 << 9,
		EN_HOWLING = 1 << 10,
		EN_DOA = 1 << 11,
		EN_WIND = 1 << 12,
		EN_AINR = 1 << 13,
		EN_AIEC = 1 << 14,
		EN_AIBF = 1 << 15,
		EN_AIECBF = 1 << 16,
		EN_SMPVP = 1 << 17,
		EN_Command = 1 << 18,
		EN_UNKWS = 1 << 19,
	} RKPreprocessEnable;

	typedef enum RkaudioRxEnable_
	{
		EN_RX_Anr = 1 << 0,
		EN_RX_HOWLING = 1 << 1,
		EN_RX_AGC = 1 << 2,
	} RkaudioRxEnable;

	typedef enum RKAUDIOExtInfo_
	{
		EXTINFO_SED_RESULT = 0,
	} RKAUDIOExtInfo;
	/*****************************************/

	/* Set the three Main Para which used to initialize the AEC,BF and RX*/
	typedef struct SKVAECParameter_ {
		int 	model_aec_en;
		float 	ref_pos;
		float 	drop_ref_channel;
		float 	delay_len;
		float 	look_ahead_array;
		float 	look_ahead_discrete;
		float 	num_array_mic;
		float*	Array_list;
		//mdf
		float 	filter_len;
		float 	auto_adapted;
		float 	adapt_rate_down;
		float 	adapt_rate_high;
		float 	reserved[10];
		//delay
		void* 	delay_para;
		char* 	filter_path;
	} SKVAECParameter;

	typedef struct SKVPreprocessParam_
	{
		/* Parameters of agc */
		int 	model_bf_en;
		float 	ref_pos;
		float 	num_ref_channel;
		float 	drop_ref_channel;
		float 	out_channel;
		float 	ainr_pos; 	//1: ainr caculater after bf;0: anir caculate in ori mic signal;
		float 	ref_pass_threshold;	// Reference mean absolute amplitude threshold for PASS detection
		float 	reserved[9];
		char*	license_path;
		void* 	fastaec_para;
		void* 	aes_para;
		void* 	aiec_para;
		void* 	dtd_para;

		void* 	bf_para;
		void* 	aibf_para;
		void* 	aiecbf_para;
		void* 	doa_para;

		void* 	anr_para;
		void* 	ainr_para;

		void* 	dereverb_para;
		void* 	agc_para;
		void* 	cng_para;
		void* 	eq_para;

		void* 	howl_para;

		void*   wakeup_para;
		void*   command_para;
		void* 	smpvp_para;
		void* 	unkws_para;
	}SKVPreprocessParam;

	typedef struct RKaudioWakeupParams_
	{
		float 	consistent_flag;
		float 	asr_score_thresh;
		float 	asr_post_smooth_length;
		float 	average_state_frames;
		float   is_run_agc;
		char*   model_path;
	}RKaudioWakeupParams;

	// typedef struct RKaudioCommandParams_
	// {
	// 	float 	consistent_flag;
	// 	float 	asr_score_thresh;
	// 	float 	asr_post_smooth_length;
	// 	float 	average_state_frames;
	// 	float   is_run_agc;
	// 	char*   model_path;
	// }RKaudioCommandParams;

	typedef struct RKaudioSimpleVoicePrintParams_
	{
		float 	target_channel_id;
		float 	voice_print_threshold1;
		float 	voice_print_threshold2;
		char*   smpvp_bin_path;
	}RKaudioSimpleVoicePrintParams;

	typedef struct RKaudioUNKWSParams_
	{
		char* 	UNKWS_encoder_path;
		char* 	UNKWS_decoder_path;
		char* 	UNKWS_joiner_path;
		char* 	keywords_path;
	}RKaudioUNKWSParams;

	typedef struct RkaudioRxParam_
	{
		/* Parameters of agc */
		int 	model_rx_en;
		void* 	anr_para;
		void* 	howl_para;
		void* 	agc_para;
	}RkaudioRxParam;

	/****************************************/
	/*The param struct of sub-mudule of AEC,BF and RX*/
	typedef struct RKAudioDelayParam_ {
		float	MaxFrame;
		float	LeastDelay;
		float	JumpFrame;
		float	DelayOffset;
		float	MicAmpThr;
		float	RefAmpThr;
		float	StartFreq;
		float	EndFreq;
		float	SmoothFactor;
	}RKAudioDelayParam;

	typedef struct SKVANRParam_ {
		float 	noiseFactor;
		float 	swU;
		float 	PsiMin;
		float 	PsiMax;
		float 	fGmin;
		float 	start_freq;
		float 	end_freq;
		float 	Sup_Freq1;
		float 	Sup_Freq2;
		float 	Sup_Energy1;
		float 	Sup_Energy2;

		float 	InterV;
		float 	BiasMin;
		float 	UpdateFrm;
		float 	NPreGammaThr;
		float 	NPreZetaThr;
		float 	SabsGammaThr0;
		float 	SabsGammaThr1;
		float 	InfSmooth;
		float 	ProbSmooth;
		float 	CompCoeff;
		float 	PrioriMin;
		float 	PostMax;
		float 	PrioriRatio;
		float 	PrioriRatioLow;
		float   SplitBand;
		float 	PrioriSmooth;
		//transient
		float 	TranMode;
		// motor
		float 	MotorBins;
		float 	MotorFactor;
		float 	MotorMute;
	} SKVANRParam;

	typedef struct RKAudioDereverbParam_
	{
		float		rlsLg;
		float     	curveLg;
		float		delay;
		float 		Dmic;
		float   	forgetting;
		float   	T60;
		float		coCoeff;
	} RKAudioDereverbParam;

	typedef struct RKAudioAESParameter_ {
		float	Beta_Up;
		float	Beta_Down;
		float	Beta_Up_Low;
		float	Beta_Down_Low;
		float	low_freq;
		float	high_freq;
		float	THD_Flag;
		float	HARD_Flag;
		float 	aphta_PD;
		float 	aphta_PD_LOW;
		float 	reserved0[10];

		float	LimitRatio[2][3];
		float	ThdSplitFreq[4][2];
		float	ThdSupDegree[4][10];
		float	reserved1[10];

		float	HardSplitFreq[5][2];
		float	HardThreshold[4];
		float	reserved2[10];
	} RKAudioAESParameter;

	typedef struct RKAudioFastaecParam_
	{
		float  filter_len; //note that, its must less aec filter len;
		float  alpha;      //EMA smoothing factor for IVM/CM (0~1), default 0.995
	} RKAudioFastaecParam;

	typedef struct RKAudioAIECParameter_ {
		float 	parallel_mode; // 0: only apply 1 channel; 1: apply all channel
		float 	post_process;	 // 0: no post porcess; 1: apply aes post process
		float 	select_chan;
		float 	continue_count;
		float 	pre_gain; //input * pre_gain;
		float 	aphta_PA;
		float 	aphta_PD;
		float 	aphta_GA;
		float 	aphta_GD;
		float 	aphta_PD_LOW;
		float 	low_freq;
		float 	high_freq;
		float 	Ratio[3];
		char*	model_path;
	} RKAudioAIECParameter;

	typedef struct RKAudioAIBFParameter_
	{
		float pre_gain;
		float mode;
		float alpha_x_ascend;
		float alpha_x_descend;
		float alpha_n_ascend;
		float alpha_n_descend;
		float post_snr_ns_th;
		float post_snr_sp_th;
		float min_gain;
		float low_freq;
		float high_freq;
		char *model_path;
	} RKAudioAIBFParameter;

	typedef struct RKAudioAIECBFParameter_
	{
		float pre_gain;
		float mode;
		float alpha_x_ascend;
		float alpha_x_descend;
		float alpha_n_ascend;
		float alpha_n_descend;
		float post_snr_ns_th;
		float post_snr_sp_th;
		float min_gain;
		float low_freq;
		float high_freq;
		char *model_path;
	} RKAudioAIECBFParameter;

	typedef struct RKAudioBFParameter_
	{
		float 	filter_size; 			// typical 512
		float 	array_type; 			// 0:Central Symmetrical Linear array; 1:Uniform Circle array;2:arbitrary array
		float 	en_winner;				// whether use winner post filter
		float 	en_weight;				// weight beamfrom output; choose one or two beamform dicide by output_channel 
		float 	en_gsc;					// whether use pf post filter
		float 	gsc_method;				// 0:LMS,1: RLS
		float 	en_fix;					// fix targ
		float 	targ1_len;				// targ1 beam_num
		float 	targ2_len;				// targ2 beam_num
		float 	*targ1;					// use when fix is on
		float 	*targ2;					// use when fix is on and out_channel = 2
		float 	bf_min_diff;
		float  	stability_threshold;
		float 	en_wind;				// special wind noise reduction
		float 	start_freq;				// starg freq to caculate BF, Hz
		float 	end_freq;				// end freq to caculate BF, it should less than sample_rate/2, Hz
		float 	input_beam_num;		// input beam num
		float 	output_beam_num;		// output beam num
		// param for weight bf, it won't use when en_fix is enable;
		float 	T_MIU;
		float 	win_low_freq;				// Hz
		float 	win_high_freq;				// Hz
		float 	win_apply_low_freq;		// Hz
		float 	win_apply_high_freq;		// Hz
		float 	low_freq;				// Hz
		float 	high_freq;				// Hz
		float 	num_array_mic;      // 阵列麦克风数（用于BF）
		float 	num_discrete_mic;   // 离散拓展麦克风数（用于chan_select）,默认离散mic排列顺序在阵列mic之后
		float 	output_mode;        // 0: BF输出, 1: chan_select输出, 2: 自动选择
		float 	chan_select_threshold; // output_mode=2时，chan_select能量超过BF的倍数阈值才切换（默认1.2）
		char*	param_path;
	}RKAudioBFParameter;

	typedef struct RKDTDParam_
	{
		float 	ksiThd_high;			 
		float 	ksiThd_low;			 
	}RKDTDParam;

	typedef struct RKAGCParam_ {
		float        	attack_time;		
		float			release_time;	
		float        	max_gain;		
		float 			max_peak;		
		float        	fRth0;			
		float        	fRk0;			
		float       	fRth1;			
		float           bits;                      
		float           frm_num;                  
		float          	attenuate_time; 
		float          	fRth2;                    
		float          	fRk1;                     
		float          	fRk2;                      
		float          	fLineGainDb;               
		float           swSmL0;                    
		float           swSmL1;                    
		float           swSmL2;                   
	} RKAGCParam;

	typedef struct RKCNGParam_
	{
		/*CNG Parameter*/
		float              fGain;                    
		float              fMpy;						
		float              fSmoothAlpha;             
		float              fSpeechGain;              
	} RKCNGParam;

	typedef struct RKaudioEqParam_ {
		float 	shwParaLen;           
		float 	pfCoeff[5][13];          
	} RKaudioEqParam;

	typedef struct RKHOWLParam_
	{
		float 	howlMode;
		float 	ratio;		// for mode4, voice changer ratio
		float 	shift_freq;	// for mode5, i.e +5, -5Hz
		float 	alpha;		// for mode6, nlp process 0 -0.5
	}RKHOWLParam;
	
	typedef struct RKDOAParam_
	{
		float 	alpha;
		float 	start_freq;
		float 	end_freq;
		float 	lg_num;			// azimuth search number
		float 	lg_pitch_num;	// pitch search number, 0 only azimuth
		float 	azimuth; 		// azimuth angle
		float 	pitch;			// pitch angle
		float   fast_search;	// only work when lg_pitch_num > 0
		float 	search_mic1;
		float 	search_mic2;
		float 	X_coord[16]; 
		float 	Y_coord[16]; 		
		float 	Z_coord[16]; 
	} RKDOAParam;

	typedef struct RKAinrParam_ {
		// 
		float			mode;		// [1, 2]
		float			datrs_optm; // [0, 1, 2]
		float			pregain;	// pregain before sent to ainr
		float			mini_gain;	// environment voice level. 0~1
		float			alpha1;		// smooth (speech ascend alpha) 
		float			alpha2;		// smooth (speech descend alpha)
		float			noise_threshold;	// if mode==1, used as noise threshold
		float			speech_threshold;	// if mode==1, used as speech threshold
		float			noise_ascend_alpha;	// if mode==1, used as noise ascend alpha
		float			noise_descend_alpha;	// if mode==1, used as noise descend alpha
		// 
		float			sfs_enable;
		float			Sup_Freq1;
		float			Sup_Freq2;
		float			Sup_Energy1;
		float			Sup_Energy2;
		float           reserved[10];
		// 
		float			eq_enable;
		float			speech_eq;
		float			eq_frqs_hz[11];
		float			eq_gains[11];
		char* 			model_path;
	}RKAinrParam;
	/*************** TX ***************/

	/* Set the Sub-Para which used to initialize the DELAY*/
	inline static void* rkaudio_delay_param_init() {
		/*RKAudioDelayParam* param = (RKAudioDelayParam*)malloc(sizeof(RKAudioDelayParam));*/
		RKAudioDelayParam* param = (RKAudioDelayParam*)calloc(1, sizeof(RKAudioDelayParam));
		param->MaxFrame = 32;          /* Maximum estimated number of delay frames */
		param->LeastDelay = 0;         /* Minimum estimated number of delay frames */
		param->JumpFrame = 12;         /* Number of frames to skip */
		param->DelayOffset = 1;        /* Delay offset in frames */
		param->MicAmpThr = 50;         /* Minimum energy threshold on the mic side */
		param->RefAmpThr = 50;         /* Minimum energy threshold on the ref side */
		param->StartFreq = 500;        /* Starting frequency of the delay estimation band */
		param->EndFreq = 4000;         /* Ending frequency of the delay estimation band */
		param->SmoothFactor = 0.99f;
		return (void*)param;
	}
	/* Set the Sub-Para which used to initialize the ANR*/
	inline static void* rkaudio_anr_param_init_tx() {
		/*SKVANRParam* param = (SKVANRParam*)malloc(sizeof(SKVANRParam));*/
		SKVANRParam* param = (SKVANRParam*)calloc(1, sizeof(SKVANRParam));
		/* anr parameters */
		//param->noiseFactor = 0.88f;//-3588.0f to compatible old json
		param->noiseFactor = 0.88f;
		param->swU = 15;
		param->PsiMin = 0.02;
		param->PsiMax = 0.516;
		param->fGmin = 0.01;
		param->start_freq = -3588;
		param->end_freq = -3588;
		param->Sup_Freq1 = -3588;
		param->Sup_Freq2 = -3588;
		param->Sup_Energy1 = 10000;
		param->Sup_Energy2 = 10000;

		param->InterV = 10;				//ANR_NOISE_EST_V
		param->BiasMin = 1.67f;			//ANR_NOISE_EST_BMIN
		param->UpdateFrm = 15;			//UPDATE_FRAME
		param->NPreGammaThr = 4.6f;		//ANR_NOISE_EST_GAMMA0
		param->NPreZetaThr = 1.67f;		//ANR_NOISE_EST_PSI0
		param->SabsGammaThr0 = 1.0f;	//ANR_NOISE_EST_GAMMA2
		param->SabsGammaThr1 = 3.0f;	//ANR_NOISE_EST_GAMMA1
		param->InfSmooth = 0.8f;		//ANR_NOISE_EST_ALPHA_S
		param->ProbSmooth = 0.7f;		//ANR_NOISE_EST_ALPHA_D
		param->CompCoeff = 1.4f;		//ANR_NOISE_EST_BETA
		param->PrioriMin = 0.0316f;		//ANR_NOISE_EST_ESP_MIN
		param->PostMax = 40.0f;			//ANR_NOISE_EST_GAMMA_MAX
		param->PrioriRatio = 0.95f;		//ANR_NOISE_EST_ALPHA
		param->PrioriRatioLow = 0.95f;	//ANR_NOISE_EST_ALPHA
		param->SplitBand = 20;
		param->PrioriSmooth = 0.7f;		//ANR_ENHANCE_BETA

		//transient
		param->TranMode = 0;			//1 << 2
		
		// motor
		param->MotorBins = 8;
		param->MotorFactor = 0.8;
		param->MotorMute = 0;
		return (void*)param;
	}
	/* Set the Sub-Para which used to initialize the Dereverb*/
	inline static void* rkaudio_dereverb_param_init() {
		/*RKAudioDereverbParam* param = (RKAudioDereverbParam*)malloc(sizeof(RKAudioDereverbParam));*/
		RKAudioDereverbParam* param = (RKAudioDereverbParam*)calloc(1, sizeof(RKAudioDereverbParam));
		param->rlsLg = 4;             /* Order of the RLS filter */
		param->curveLg = 20;          /* Order of the distribution curve */
		param->delay = 2;             /* Delay of the RLS filter */
		param->forgetting = 0.98;     /* Forgetting factor of the RLS filter */
		param->T60 = 0.3; // 1.5;     /* Estimated reverberation time (in seconds); the larger the value, the stronger the dereverberation ability, but it increases the risk of over-suppression */
		param->coCoeff = 1;           /* Coherence adjustment coefficient to prevent over-suppression; higher values increase suppression strength. Recommended range: between 0.5 and 2 */

		return (void*)param;
	}
	inline static void* rkaudio_fastaec_param_init() {
		/*RKAudioDereverbParam* param = (RKAudioDereverbParam*)malloc(sizeof(RKAudioDereverbParam));*/
		RKAudioFastaecParam* param = (RKAudioFastaecParam*)calloc(1, sizeof(RKAudioFastaecParam));
		param->filter_len = 4;             /* must less than mdf filter len */
		param->alpha = 0.995f;             /* EMA smoothing factor, larger = slower adaption, more stable */
		return (void*)param;
	}

	/* Set the Sub-Para which used to initialize the AES*/
	inline static void* rkaudio_aes_param_init() {
		/*RKAudioAESParameter* param = (RKAudioAESParameter*)malloc(sizeof(RKAudioAESParameter));*/
		RKAudioAESParameter* param = (RKAudioAESParameter*)calloc(1, sizeof(RKAudioAESParameter));
		//param->Beta_Up = 0.002f; /* increase speed, -3588.0f to compatible old json*/
		param->Beta_Up = 0.0003f;
		param->Beta_Down = 0.0003f; /* decrease speed */
		param->Beta_Up_Low = 0.0003f; /* low frequency increase speed */
		param->Beta_Down_Low = 0.0003f; /* ow frequency decrease speed */
		param->aphta_PD = 0.2f;
		param->aphta_PD_LOW = 0.08f;
		param->low_freq = 600;
		param->high_freq = 3750;
		param->THD_Flag = 0;	/* 1 open THD, 0 close THD */
		param->HARD_Flag = 0;	/* 1 open Hard Suppress, 0 close Hard Suppress */
		int i, j;
		for (i = 0; i < 2; i++)
			for (j = 0; j < 3; j++)
				param->LimitRatio[i][j] = LimitRatio[i][j];
		for (i = 0; i < 4; i++)
			for (j = 0; j < 2; j++)
				param->ThdSplitFreq[i][j] = ThdSplitFreq[i][j];
		for (i = 0; i < 4; i++)
			for (j = 0; j < 10; j++)
				param->ThdSupDegree[i][j] = ThdSupDegree[i][j];
		
		for (i = 0; i < 5; i++)
			for (j = 0; j < 2; j++)
				param->HardSplitFreq[i][j] = HardSplitFreq[i][j];
		for (i = 0; i < 4; i++)
			param->HardThreshold[i] = HardThreshold[i];
		return (void*)param;
	}

	inline static void* rkaudio_aiec_param_init() {
		RKAudioAIECParameter* param = (RKAudioAIECParameter*)calloc(1, sizeof(RKAudioAIECParameter));
		//param->Beta_Up = 0.002f; /* increase speed, -3588.0f to compatible old json*/
		param->parallel_mode = 0;	// 0: only apply 1 channel; 1: apply all channel
		param->post_process = 0; // 0: no post porcess; 1: apply aes post process 
		param->select_chan = 0;
		param->continue_count = 10; 
		param->pre_gain = 5.0;
		param->aphta_PA = 0.5f;
		param->aphta_PD = 0.2f;
		param->aphta_PD_LOW = 0.08f;
		param->low_freq = 600;
		param->high_freq = 3750;
		int i;
		for (i = 0; i < 3; i++)
			param->Ratio[i] = Ratio[i];

		param->model_path = (char*)"/home/lmf/Aec_bf_combine/include/model_files/aiec_model_files/mnn_models/Scnet_Mask_Aec_160M_0910.bin"; 
		return (void*)param;
	}

	static float targ1[2] = {0};
	static float targ2[2] = {3};
	inline static void* rkaudio_bf_param_init() {
		RKAudioBFParameter* param = (RKAudioBFParameter*)calloc(1, sizeof(RKAudioBFParameter));
		param->array_type = 0;			// 0:Central Symmetrical Linear array; 1:Uniform Circle array;2:arbitrary array
		param->en_winner = 1;			// whether enable winner
		param->en_weight = 1;			// whether enable weight
		param->en_gsc = 0;
		param->gsc_method = 0;
		param->en_fix = 0;				// fix targ
		param->targ1_len = 1;
		param->targ2_len = 1;
		param->targ1 = targ1;				// use when fix is on
		param->targ2 = targ2;				// use when fix is on
		param->bf_min_diff = 1;
		param->stability_threshold = 3;
		param->en_wind = 0;
		param->filter_size = 512;
		param->start_freq = 0;			//Hz
		param->end_freq = 8000; 		//Hz
		param->input_beam_num = 5;
		param->output_beam_num = 5;
		param->T_MIU = 0.35;
		param->win_low_freq = 800;			//Hz
		param->win_high_freq = 4000;		//Hz
		param->low_freq = 1500;			//Hz
		param->high_freq = 5000;		//Hz
		param->num_array_mic = 8;
		param->num_discrete_mic = 4;
		param->output_mode = 0;		// 0: BF输出, 1: chan_select输出, 2: 自动选择
		param->chan_select_threshold = 1.2f;
		param->param_path = (char*)"/home/lmf/Aec_bf_combine/code/lib/model_files/aibf_model_files/mnn_models/bf_param/wopt_8mic_linear_UUmm.bin"; //load bf param
		return (void*)param;
	}

	inline static void* rkaudio_aibf_param_init() {
		RKAudioAIBFParameter* param = (RKAudioAIBFParameter*)calloc(1, sizeof(RKAudioAIBFParameter));
		param->pre_gain = 1.0;
		param->alpha_x_ascend = 0.9;
		param->alpha_x_descend = 0.6;
		param->alpha_n_ascend = 0.2;
		param->alpha_n_descend = 0.9;
		param->post_snr_ns_th = 0.5;
		param->post_snr_sp_th = 2.0;
		param->min_gain = 0.1;
		param->low_freq = 0;
		param->high_freq = 8000;
		param->model_path = (char*)"/home/lmf/Aec_bf_combine/code/lib/model_files/aibf_model_files/mnn_models/SCNet_Mask_Bf_160M_1112.bin"; 
		return (void*)param;
	}

	/* Set the Sub-Para which used to initialize the DTD*/
	inline static void* rkaudio_dtd_param_init()
	{
		/*RKDTDParam* param = (RKDTDParam*)malloc(sizeof(RKDTDParam));*/
		RKDTDParam* param = (RKDTDParam*)calloc(1, sizeof(RKDTDParam));
		/* dtd paremeters*/
		param->ksiThd_high = 0.60f;										            /* dtd threshold */
		param->ksiThd_low = 0.50f;
		return (void*)param;
	}
	/* Set the Sub-Para which used to initialize the AGC*/
	inline static void* rkaudio_agc_param_init()
	{
		/*RKAGCParam* param = (RKAGCParam*)malloc(sizeof(RKAGCParam));*/
		RKAGCParam* param = (RKAGCParam*)calloc(1, sizeof(RKAGCParam));
		/* New AGC parameters */
		param->attack_time = 200.0;      /* Attack time, i.e., the time required for AGC gain to increase */
		param->release_time = 400.0;     /* Release time, i.e., the time required for AGC gain to decrease */
		//param->max_gain = 35.0;        /* Maximum gain, also the gain for the linear section, in dB */
		param->max_gain = 20;            /* Maximum gain, also the gain for the linear section, in dB */
		param->max_peak = -3.0;          /* Maximum output energy after AGC processing, range in dB */
		param->fRk0 = 2;                /* Slope of the expansion section */
		param->fRth2 = -30;             /* Compression section start energy dB threshold, also the end threshold of the linear section. The gain gradually decreases. Note: fRth2 + max_gain < max_peak */
		param->fRth1 = -45;             /* Expansion section end energy dB threshold, also the start threshold of the linear section. Energy above this region is amplified by max_gain */
		param->fRth0 = -50;             /* Noise gate threshold */

		/* Invalid parameters */
		param->bits = 16;					/* bit rate */
		param->frm_num = 4;            /* sub frame num */
		param->attenuate_time = 200;   /* Noise attenuation time, i.e., the time needed for noise segment gain to decay to 1 */
		param->fRk1 = 0.8;              /* Slope of the expansion section */
		param->fRk2 = 0.4;              /* Slope of the expansion section */
		param->fLineGainDb = -25.0f;    /* No gain applied below this value during the initial attenuate_time (ms) */
		param->swSmL0 = 1;             /*  this misc flag is bitmap, bit0: more_grow_protect bit1: fast_update_gain */
		param->swSmL1 = 80;             /* Number of time-domain smoothing points in the linear section */
		param->swSmL2 = 80;             /* Number of time-domain smoothing points in the compression section */


		return (void*)param;
	}
	/* Set the Sub-Para which used to initialize the CNG*/
	inline static void* rkaudio_cng_param_init()
	{
		/*RKCNGParam* param = (RKCNGParam*)malloc(sizeof(RKCNGParam));*/
		RKCNGParam* param = (RKCNGParam*)calloc(1, sizeof(RKCNGParam));
		/* cng paremeters */
		param->fSmoothAlpha = 0.99f;                                        /* INT16 Q15 Smoothing factor applied to comfort noise */
		param->fSpeechGain = 0;                                             /* INT16 Q15 Degree of speech texture simulation applied to comfort noise */
		param->fGain = 10.0;                                                /* INT16 Q0 Amplitude ratio applied to comfort noise */
		param->fMpy = 10;                                                   /* INT16 Q0 Amplitude for white noise random number generation */
		return (void*)param;
	}
	/* Set the Sub-Para which used to initialize the EQ*/
	inline static void* rkaudio_eq_param_init() {
		/*RKaudioEqParam* param = (RKaudioEqParam*)malloc(sizeof(RKaudioEqParam));*/
		RKaudioEqParam* param = (RKaudioEqParam*)calloc(1, sizeof(RKaudioEqParam));
		param->shwParaLen = 65;
		int i, j;
		for (i = 0; i < 5; i++) {
			for (j = 0; j < 13; j++) {
				param->pfCoeff[i][j] = EqPara_16k[i][j];
			}
		}
		return (void*)param;
	}
	/* Set the Sub-Para which used to initialize the HOWL*/
	inline static void* rkaudio_howl_param_init_tx() {
		/*RKHOWLParam* param = (RKHOWLParam*)malloc(sizeof(RKHOWLParam));*/
		RKHOWLParam* param = (RKHOWLParam*)calloc(1, sizeof(RKHOWLParam));
		param->howlMode = 0;	//0b110000
		param->ratio = 0.95;			 //vc changer ratio,for mode4
		param->shift_freq = -10;	//decrease 10Hz,for mode5
		param->alpha = 0.3;			//nlp alpha, for mode6
		return (void*)param;
	}

	static float X_cod[8] = {0.035, 0.0247, 0, -0.0247, -0.035, -0.0247, 0, 0.0247};
	static float Y_cod[8] = {0, 0.0247, 0.035, 0.0247, 0, -0.0247, -0.035, -0.0247};
	static float Z_cod[8] = {0, 0, 0, 0, 0, 0, 0, 0};

	inline static void* rkaudio_doa_param_init() {
		RKDOAParam* param = (RKDOAParam*)calloc(1, sizeof(RKDOAParam));
		param->alpha = 0.9;
		param->start_freq = 600;
		param->end_freq = 3000;
		param->lg_num = 60;
		param->lg_pitch_num = 0; 
		param->azimuth = 360;
		param->pitch = 180;
		param->fast_search = 0;
		param->search_mic1 = 3;
		param->search_mic1 = 4;
		for (int i = 0; i < 8; i++){
			param->X_coord[i] = X_cod[i];
			param->Y_coord[i] = Y_cod[i];
			param->Z_coord[i] = Z_cod[i];
		}
		
		return (void*)param;
	}

	inline static void* rkaudio_smpvp_param_init()
	{
		RKaudioSimpleVoicePrintParams* param = (RKaudioSimpleVoicePrintParams*)calloc(1, sizeof(RKaudioSimpleVoicePrintParams));
		param->target_channel_id = 0;
		param->voice_print_threshold1 = 9.5;
		param->voice_print_threshold2 = 11.0;
		param->smpvp_bin_path=(char*)"/mnt/e/rockchip_4mic_aec_beamform/model_files/smpvp_files/codebook.bin";
		return (void*)param;
	}

	inline static void* rkaudio_unkws_param_init()
	{
		RKaudioUNKWSParams* param = (RKaudioUNKWSParams*)calloc(1, sizeof(RKaudioUNKWSParams));
		param->UNKWS_encoder_path = (char*)"/mnt/e/rockchip_4mic_aec_beamform/model_files/unkws_model_files/mnn_model/unkws_encoder_mnn.bin";
		param->UNKWS_decoder_path = (char*)"/mnt/e/rockchip_4mic_aec_beamform/model_files/unkws_model_files/mnn_model/unkws_decoder_mnn.bin";
		param->UNKWS_joiner_path = (char*)"/mnt/e/rockchip_4mic_aec_beamform/model_files/unkws_model_files/mnn_model/unkws_joiner_mnn.bin";
		param->keywords_path = (char*)"/mnt/e/rockchip_4mic_aec_beamform/model_files/unkws_model_files/keywords.txt";
		return (void*)param;
	}

	inline static void* rkaudio_ainr_param_init() {
		RKAinrParam* param = (RKAinrParam*)calloc(1, sizeof(RKAinrParam));
		param->mode = 1;				// mode. 1, 2
		param->datrs_optm = 0;			// optimization mode of data transfer, default 2.
		param->alpha1 = 0.2;			// smooth (speech ascend alpha)
		param->alpha2 = 0.2;			// smooth (speech descend alpha)
		param->mini_gain = 0.05;			// environment voice level. 0~1
		param->pregain = 1;				// pregain before sent to ainr
		// preserve
		param->noise_threshold = 0.9;			// if mode==1, used as noise threshold
		param->speech_threshold = 5.0;			// if mode==1, used as speech threshold
		param->noise_ascend_alpha = 0.2;			// if mode==1, used as noise ascend alpha
		param->noise_descend_alpha = 0.2;		// if mode==1, used as noise descend alpha
		// 
		param->sfs_enable = 0;
		param->Sup_Freq1 = -3588;
		param->Sup_Freq2 = -3588;
		param->Sup_Energy1 = 10000;
		param->Sup_Energy2 = 10000;
		// 
		param->eq_enable = 0;
		param->speech_eq = 0;

		float eq_frqs_hz[11] = {0,	1000,	2400,	4000,	6000,	7500, 8000, -1, -1, -1, -1 };
		float eq_gains[11] = {0.6,  0.8,	0.8,	1.25,	1.25,	1.0,  1.0,	0,	0, 0, 0 };
		int i;
		for (i = 0; i < 11; i++) {
			param->eq_frqs_hz[i] = eq_frqs_hz[i];
			param->eq_gains[i] = eq_gains[i];
		}
		param->model_path = (char*)"/home/lmf/Aec_bf_combine/code/lib/model_files/ainr_model_files/mnn_models/168M/DPCRNbs_168M_2frm_2508041708_E080_fp.bin";
		return (void*)param;
	}
	/************* RX *************/
	inline static void* rkaudio_anr_param_init_rx() {
		/*SKVANRParam* param = (SKVANRParam*)malloc(sizeof(SKVANRParam));*/
		SKVANRParam* param = (SKVANRParam*)calloc(1, sizeof(SKVANRParam));
		/* anr parameters */
		param->noiseFactor = 0.88f;
		param->swU = 12;
		param->PsiMin = 0.02;
		param->PsiMax = 0.516;
		param->fGmin = 0.05;

		param->Sup_Freq1 = -3588;
		param->Sup_Freq2 = -3588;
		param->Sup_Energy1 = 100000;
		param->Sup_Energy2 = 100000;

		param->InterV = 8;				//ANR_NOISE_EST_V
		param->BiasMin = 1.67f;			//ANR_NOISE_EST_BMIN
		param->UpdateFrm = 15;			//UPDATE_FRAME
		param->NPreGammaThr = 4.6f;		//ANR_NOISE_EST_GAMMA0
		param->NPreZetaThr = 1.67f;		//ANR_NOISE_EST_PSI0
		param->SabsGammaThr0 = 1.0f;	//ANR_NOISE_EST_GAMMA2
		param->SabsGammaThr1 = 3.0f;	//ANR_NOISE_EST_GAMMA1
		param->InfSmooth = 0.8f;		//ANR_NOISE_EST_ALPHA_S
		param->ProbSmooth = 0.7f;		//ANR_NOISE_EST_ALPHA_D
		param->CompCoeff = 1.4f;		//ANR_NOISE_EST_BETA
		param->PrioriMin = 0.0316f;		//ANR_NOISE_EST_ESP_MIN
		param->PostMax = 40.0f;			//ANR_NOISE_EST_GAMMA_MAX
		param->PrioriRatio = 0.95f;		//ANR_NOISE_EST_ALPHA
		param->PrioriRatioLow = 0.95f;	//ANR_NOISE_EST_ALPHA
		param->SplitBand = 20;
		param->PrioriSmooth = 0.7f;		//ANR_ENHANCE_BETA

		//transient
		param->TranMode = 0;

		// motor
		param->MotorBins = 8;
		param->MotorFactor = 0.8;
		param->MotorMute = 0;
		return (void*)param;
	}
	inline static void* rkaudio_howl_param_init_rx() {
		/*RKHOWLParam* param = (RKHOWLParam*)malloc(sizeof(RKHOWLParam));*/
		RKHOWLParam* param = (RKHOWLParam*)calloc(1, sizeof(RKHOWLParam));
		param->howlMode = 0;	//
		param->ratio = 0.95;			 //vc changer ratio,for mode4
		param->shift_freq = -10;	//decrease 10Hz,for mode5
		param->alpha = 0.3;			// nlp process, for mode6
		return (void*)param;
	}
	inline static void* rkaudio_agc_param_init_rx()
	{
		RKAGCParam* param = (RKAGCParam*)malloc(sizeof(RKAGCParam));

		param->attack_time = 200.0;		
		param->release_time = 200.0;	
		//param->max_gain = 35.0;		
		param->max_gain = 20.0;			
		param->max_peak = -1;			
		param->fRk0 = 2;				
		param->fRth2 = -25;				
		param->fRth1 = -60;				
		param->fRth0 = -65;				
		/* Invalid parameters */
		param->bits = 16000;
		param->frm_num = 4;                   
		param->attenuate_time = 1000; 
		param->fRk1 = 0.8;                      
		param->fRk2 = 0.4;                      
		param->fLineGainDb = -25.0f;               
		param->swSmL0 = 0;                    
		param->swSmL1 = 80;                    
		param->swSmL2 = 80;                    

		return (void*)param;
	}
	/* Set the Sub-Para which used to initialize the AEC*/
	inline static void* rkaudio_aec_param_init()
	{
		/*SKVAECParameter* param = (SKVAECParameter*)malloc(sizeof(SKVAECParameter));*/
		SKVAECParameter* param = (SKVAECParameter*)calloc(1, sizeof(SKVAECParameter));
		param->ref_pos = REF_POSITION;
		param->drop_ref_channel = NUM_DROP_CHANNEL;
		param->model_aec_en = EN_DELAY;		//param->model_aec_en = EN_DELAY;
		param->delay_len = 0;						//-3588 to compatible old json
		//param->delay_len = -3588;
		param->look_ahead_array = 0;
		param->look_ahead_discrete = 0;
		param->num_array_mic = 0;
		param->Array_list = Array;
		//mdf
		param->filter_len = 4;
		param->auto_adapted = 1;
		param->adapt_rate_down = 0.05;
		param->adapt_rate_high = 0.25;
		//delay
		param->delay_para = rkaudio_delay_param_init();
		//param->filter_path = (char*)"C:/works/customer/�/uac/mdf.bin";
		param->filter_path = NULL;
		return (void*)param;
	}
	/* Set the Sub-Para which used to initialize the BF*/
	inline static void* rkaudio_preprocess_param_init()
	{
		SKVPreprocessParam* param = (SKVPreprocessParam*)calloc(1, sizeof(SKVPreprocessParam));
		//param->model_bf_en = EN_Fix | EN_HOWLING | EN_Anr | EN_Dereverberation | EN_Agc;
		//param->model_bf_en = EN_Fix | EN_HOWLING | EN_Anr | EN_Dereverberation | EN_Agc;
		param->model_bf_en = EN_AIBF;
		//param->model_bf_en = EN_Fix | EN_Agc | EN_HOWLING | EN_Anr | EN_Dereverberation;
		//param->model_bf_en = EN_Anr; // | EN_Wakeup | EN_WIND | EN_Dereverberation | EN_HOWLING;
		//param->model_bf_en = EN_Fastaec | EN_AES | EN_Dereverberation | EN_Agc;
		//param->model_bf_en = EN_Fix | EN_Agc | EN_AINR | EN_Dereverberation;
		//param->model_bf_en = EN_DOA | EN_Fix | EN_AINR | EN_Anr ;
		//param->model_bf_en = EN_Fastaec | EN_Anr | EN_AES | EN_Agc;
		//param->model_bf_en = EN_Fastaec | EN_Agc;
		//param->model_bf_en = EN_Fastaec | EN_Fix | EN_Agc | EN_Anr;
		//param->model_bf_en = EN_Fastaec | EN_AES | EN_Agc | EN_Fix | EN_Anr | EN_HOWLING;
		//param->model_bf_en = EN_Fastaec | EN_AES | EN_Anr | EN_Agc;
		param->ref_pos = REF_POSITION;
		param->num_ref_channel = NUM_REF_CHANNEL;
		param->drop_ref_channel = NUM_DROP_CHANNEL;
		param->out_channel = 2;		
		param->ainr_pos = 0;
		param->ref_pass_threshold = 30.0f;
		param->license_path = (char*)"/home/lmf/license_data.bin";
		param->bf_para = rkaudio_bf_param_init();
		param->anr_para = rkaudio_anr_param_init_tx();
		param->dereverb_para = rkaudio_dereverb_param_init();
		param->aes_para = rkaudio_aes_param_init();
		param->fastaec_para = rkaudio_fastaec_param_init();
		param->dtd_para = rkaudio_dtd_param_init();
		param->agc_para = rkaudio_agc_param_init();
		param->cng_para = rkaudio_cng_param_init();
		param->eq_para = rkaudio_eq_param_init();
		param->howl_para = rkaudio_howl_param_init_tx();
		param->doa_para = rkaudio_doa_param_init();
		param->ainr_para = rkaudio_ainr_param_init();
		param->aiec_para = rkaudio_aiec_param_init();
		param->aibf_para = rkaudio_aibf_param_init();
		param->smpvp_para = rkaudio_smpvp_param_init();
		param->unkws_para = rkaudio_unkws_param_init();
		return (void*)param;
	}
	/* Set the Sub-Para which used to initialize the RX*/
	inline static void* rkaudio_rx_param_init()
	{
		//RkaudioRxParam* param = (RkaudioRxParam*)malloc(sizeof(RkaudioRxParam));
		RkaudioRxParam* param = (RkaudioRxParam*)calloc(1, sizeof(RkaudioRxParam));
		param->model_rx_en = EN_RX_AGC;
		param->anr_para = rkaudio_anr_param_init_rx();
		param->howl_para = rkaudio_howl_param_init_rx();
		param->agc_para = rkaudio_agc_param_init_rx();
		return (void*)param;
	}
	typedef struct RKAUDIOParam_
	{
		int model_en;
		void* aec_param;
		void* bf_param;
		void* rx_param;
		float read_size;
	} RKAUDIOParam;

	inline static void rkaudio_aec_param_destory(void* param_)
	{
		SKVAECParameter* param = (SKVAECParameter*)param_;
		free(param->delay_para); param->delay_para = NULL;
		free(param); param = NULL;
	}

	inline static void rkaudio_preprocess_param_destory(void* param_)
	{
		SKVPreprocessParam* param = (SKVPreprocessParam*)param_;
		
		free(param->aes_para); param->aes_para = NULL;
		free(param->aiec_para); param->aiec_para = NULL;
		free(param->fastaec_para); param->fastaec_para = NULL;
		free(param->dtd_para); param->dtd_para = NULL;

		free(param->anr_para); param->anr_para = NULL;
		free(param->ainr_para); param->ainr_para = NULL;

		free(param->bf_para); param->bf_para = NULL;
		free(param->aibf_para); param->aibf_para = NULL;
		free(param->doa_para); param->doa_para = NULL;

		free(param->agc_para); param->agc_para = NULL;
		free(param->cng_para); param->cng_para = NULL;
		free(param->eq_para); param->eq_para = NULL;
		free(param->howl_para); param->howl_para = NULL;
		free(param->dereverb_para); param->dereverb_para = NULL;

		free(param->smpvp_para); param->smpvp_para = NULL;
		free(param->unkws_para); param->unkws_para = NULL;
		free(param); param = NULL;
	}

	inline static void rkaudio_rx_param_destory(void* param_)
	{
		RkaudioRxParam* param = (RkaudioRxParam*)param_;
		free(param->anr_para); param->anr_para = NULL;
		free(param->howl_para); param->howl_para = NULL;
		free(param->agc_para); param->agc_para = NULL;
		free(param); param = NULL;
	}

	inline static void rkaudio_param_deinit(void* param_)
	{
		RKAUDIOParam* param = (RKAUDIOParam*)param_;
		if (param->aec_param != NULL)
			rkaudio_aec_param_destory(param->aec_param);
		if (param->bf_param != NULL)
			rkaudio_preprocess_param_destory(param->bf_param);
		if (param->rx_param != NULL)
			rkaudio_rx_param_destory(param->rx_param);
	}

	void* rkaudio_preprocess_init(int rate, int bits, int src_chan, int ref_chan, RKAUDIOParam* param);
	void* rkaudio_preprocess_init_by_conf(int rate, int bits, int src_chan, int ref_chan, char *conf_path);
	void rkaudio_param_printf(int src_chan, int ref_chan, RKAUDIOParam* param);
	int rkaudio_Doa_invoke(void* st_ptr);
	int rkaudio_Cir_Doa_invoke(void* st_ptr, int* ang_doa, int* pth_doa);

	int rkaudio_out_chan_invoke(void* st_ptr);

	int rkaudio_bf_targ_invoke(void* st_ptr, int* bf_targ1, int* bf_targ2);

	int rkaudio_preprocess_get_cmd_id(void* st_ptr, float* cmd_score, int* cmd_id);

	int rkaudio_preprocess_get_asr_id(void* st_ptr, float* asr_score, int* asr_id);

	char* rkaudio_preprocess_get_vpm_id(void* st_ptr,char* vpm_id);

	char* get_triggered_keyword_rkaudiopreprocess(void* st_);

	void* rkaudio_param_get(void* st_ptr);
	int rkaudio_param_set(void* st_ptr, void* param);

	void rkaudio_preprocess_destory(void* st_ptr);

	int rkaudio_preprocess_short(void* st_ptr, short* in, short* out, int in_size, int* wakeup_status);

	int rkaudio_preprocess_float(void* st_ptr, float** in, float** out, int in_size, int* wakeup_status);

	int rkaudio_rx_short(void* st_ptr, short* in, short* out);

	int rkaudio_mdf_dump(void* st_ptr, short* out);

	void rkaudio_param_deinit(void* param_);

	int rkaudio_extinfo_get(void* st_ptr, int type, void *info);
	int rkaudio_extinfo_set(void* st_ptr, int type, int info);

	int rkaudio_ainr_set_level(void* st_ptr, int level);
	int rkaudio_ainr_set_enable(void* st_ptr, int enable);

	void delete_codebook_api_rkaudiopreprocess(void* st_, char* ID_person, int ID_audio);
	void set_vpm_flag_rkaudiopreprocess(void* st_, int flag); //match voice_print:1   enroll voice_print:0
	void set_vpm_IDperson_rkaudiopreprocess(void* st_, char* ID_person);
	void save_codebook_to_bin_rkaudiopreprocess(void* st_);
	void update_codebook_from_bin_rkaudiopreprocess(void* st_);
	void print_list_code_book_person_rkaudiopreprocess(void* st_);

#ifdef __cplusplus
}
#endif

#endif    // _RKAUDIO_PREPROCESS_H_
