#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "weights.h"

#include "parameters.h"
#include "input.h"
#ifdef USE_NNOM
#include "nnom.h"
#endif

q7_t conv1_out[45*45*32];
q7_t pool1_out[23*23*32];
q7_t conv2_out[23*23*32];
q7_t pool2_out[12*12*16];
q7_t conv3_out[12*12*16];
q7_t pool3_out[6*6*16];

q7_t fc1_out[10];
q7_t y_out[10];
#if IMG == 0 
    #define IMG_DATA IMAGE_DATA_199_0
    #define INPUT_Q INPUT_199_0_Q
#elif IMG == 1
    #define IMG_DATA IMAGE_DATA_290_1
    #define INPUT_Q INPUT_290_1_Q 
#elif IMG == 2 
    #define IMG_DATA IMAGE_DATA_214_2
    #define INPUT_Q INPUT_214_2_Q
#elif IMG == 3
    #define IMG_DATA IMAGE_DATA_106_3
    #define INPUT_Q INPUT_106_3_Q
#elif IMG == 4
    #define IMG_DATA IMAGE_DATA_15_4
    #define INPUT_Q INPUT_15_4_Q
#elif IMG == 5
    #define IMG_DATA IMAGE_DATA_2_5
    #define INPUT_Q INPUT_2_5_Q
#elif IMG == 6
    #define IMG_DATA IMAGE_DATA_0_6
    #define INPUT_Q INPUT_0_6_Q
#elif IMG == 7
    #define IMG_DATA IMAGE_DATA_9_7
    #define INPUT_Q INPUT_9_7_Q
#elif IMG == 8
    #define IMG_DATA IMAGE_DATA_5_8
    #define INPUT_Q INPUT_5_8_Q
#elif IMG == 9
    #define IMG_DATA IMAGE_DATA_12_9
    #define INPUT_Q INPUT_12_9_Q
#elif IMG == 10
    #define IMG_DATA IMAGE_DATA_150_0
    #define INPUT_Q INPUT_150_0_Q
#elif IMG == 11
    #define IMG_DATA IMAGE_DATA_10_1
    #define INPUT_Q INPUT_10_1_Q 
#elif IMG == 12
    #define IMG_DATA IMAGE_DATA_200_2
    #define INPUT_Q INPUT_200_2_Q
#elif IMG == 13
    #define IMG_DATA IMAGE_DATA_13_3
    #define INPUT_Q INPUT_13_3_Q
#elif IMG == 14
    #define IMG_DATA IMAGE_DATA_15_4
    #define INPUT_Q INPUT_15_4_Q
#elif IMG == 15
    #define IMG_DATA IMAGE_DATA_19_5
    #define INPUT_Q INPUT_19_5_Q
#elif IMG == 16
    #define IMG_DATA IMAGE_DATA_1_6
    #define INPUT_Q INPUT_1_6_Q
#elif IMG == 17
    #define IMG_DATA IMAGE_DATA_100_7
    #define INPUT_Q INPUT_100_7_Q
#elif IMG == 18
    #define IMG_DATA IMAGE_DATA_250_8
    #define INPUT_Q INPUT_250_8_Q
#elif IMG == 19
    #define IMG_DATA IMAGE_DATA_14_9
    #define INPUT_Q INPUT_14_9_Q
#endif

int main(int argc, char* argv[])
{
	printf("loading input&weights...\nImg choosenn = %d\n",IMG);
	printf("Expected classification %d...\n", expctd_lbl[IMG]);

	#define CONV1_IM_DIM 45 
	#define CONV1_IM_CH  1
	#define CONV1_OUT_CH 32
	#define CONV1_KER_DIM 5
	#define CONV1_PADDING ((CONV1_KER_DIM-1)/2)
	#define CONV1_STRIDE  1
	#define CONV1_OUT_DIM 45 
	#define CONV1_BIAS_LSHIFT (INPUT_Q+W_CONV1_Q-B_CONV1_Q)
	#define CONV1_OUT_RSHIFT  (INPUT_Q+W_CONV1_Q-CONV1_OUT_Q)
#ifndef USE_NNOM
    q7_t input[CONV1_IM_DIM * CONV1_IM_DIM] = IMG_DATA;
    q7_t W_conv1[CONV1_IM_CH * CONV1_KER_DIM * CONV1_KER_DIM * CONV1_OUT_CH] = W_CONV1;
    
    q7_t b_conv1[CONV1_OUT_CH] = B_CONV1;
	if (ARM_MATH_SUCCESS !=arm_convolve_HWC_q7_basic(input, CONV1_IM_DIM, CONV1_IM_CH, W_conv1, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING,
						  CONV1_STRIDE, b_conv1, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, conv1_out, CONV1_OUT_DIM,
						  (q15_t*)pool1_out, NULL)) {
	printf("CONV1 failed...\n");
                          }

	arm_relu_q7(conv1_out, CONV1_OUT_DIM * CONV1_OUT_DIM * CONV1_OUT_CH);
#endif
	#define POOL1_KER_DIM 2
	#define POOL1_PADDING ((POOL1_KER_DIM-1)/2)
	#define POOL1_STRIDE  2
	#define POOL1_OUT_DIM 23 
#ifndef USE_NNOM
	arm_maxpool_q7_HWC(conv1_out, CONV1_OUT_DIM, CONV1_OUT_CH, POOL1_KER_DIM,
						  POOL1_PADDING, POOL1_STRIDE, POOL1_OUT_DIM, NULL, pool1_out);
#endif
	#define CONV2_IM_DIM 23
	#define CONV2_IM_CH  32
	#define CONV2_OUT_DIM 23
	#define CONV2_OUT_CH 32
	#define CONV2_STRIDE 1
	#define CONV2_KER_DIM 5
	#define CONV2_PADDING ((CONV2_KER_DIM-1)/2)
	#define CONV2_BIAS_LSHIFT (CONV1_OUT_Q+W_CONV2_Q-B_CONV2_Q)
	#define CONV2_OUT_RSHIFT  (CONV1_OUT_Q+W_CONV2_Q-CONV2_OUT_Q)
#ifndef USE_NNOM
    q7_t W_conv2[CONV2_IM_CH * CONV2_KER_DIM * CONV2_KER_DIM * CONV2_OUT_CH] = W_CONV2;
    q7_t b_conv2[CONV2_OUT_CH] = B_CONV2;
	if (ARM_MATH_SUCCESS !=arm_convolve_HWC_q7_fast(pool1_out, CONV2_IM_DIM, CONV2_IM_CH, W_conv2, CONV2_OUT_CH, CONV2_KER_DIM,
						  CONV2_PADDING, CONV2_STRIDE, b_conv2, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, conv2_out,
						  CONV2_OUT_DIM, (q15_t*)conv1_out, NULL)) {
	printf("CONV2 failed...\n");
                          }

	arm_relu_q7(conv2_out, CONV2_OUT_DIM * CONV2_OUT_DIM * CONV2_OUT_CH);
#endif
	#define POOL2_KER_DIM 2
	#define POOL2_PADDING ((POOL2_KER_DIM-1)/2)
	#define POOL2_STRIDE  2
	#define POOL2_OUT_DIM 12 
#ifndef USE_NNOM
	arm_maxpool_q7_HWC(conv2_out, CONV2_OUT_DIM, CONV2_OUT_CH, POOL2_KER_DIM,
						  POOL2_PADDING, POOL2_STRIDE, POOL2_OUT_DIM, NULL, pool2_out);
#endif

	#define CONV3_IM_DIM 12
	#define CONV3_IM_CH  32
	#define CONV3_OUT_DIM 12
	#define CONV3_OUT_CH 16
	#define CONV3_STRIDE 1
	#define CONV3_KER_DIM 5
	#define CONV3_PADDING ((CONV3_KER_DIM-1)/2)
	#define CONV3_BIAS_LSHIFT (CONV2_OUT_Q+W_CONV3_Q-B_CONV3_Q)
	#define CONV3_OUT_RSHIFT  (CONV2_OUT_Q+W_CONV3_Q-CONV3_OUT_Q)
#ifndef USE_NNOM
    q7_t W_conv3[CONV3_IM_CH * CONV3_KER_DIM * CONV3_KER_DIM * CONV3_OUT_CH] = W_CONV3;
    q7_t b_conv3[CONV3_OUT_CH] = B_CONV3;
	if (ARM_MATH_SUCCESS !=arm_convolve_HWC_q7_fast(pool2_out, CONV3_IM_DIM, CONV3_IM_CH, W_conv3, CONV3_OUT_CH, CONV3_KER_DIM,
						  CONV3_PADDING, CONV3_STRIDE, b_conv3, CONV3_BIAS_LSHIFT, CONV3_OUT_RSHIFT, conv3_out,
						  CONV3_OUT_DIM, (q15_t*)conv1_out, NULL)) {
	printf("CONV3 failed...\n");
                          }

	arm_relu_q7(conv3_out, CONV3_OUT_DIM * CONV3_OUT_DIM * CONV3_OUT_CH);
#endif
	#define POOL3_KER_DIM 2
	#define POOL3_PADDING ((POOL3_KER_DIM-1)/2)
	#define POOL3_STRIDE  2
	#define POOL3_OUT_DIM 6 
#ifndef USE_NNOM
	arm_maxpool_q7_HWC(conv3_out, CONV3_OUT_DIM, CONV3_OUT_CH, POOL3_KER_DIM,
						  POOL3_PADDING, POOL3_STRIDE, POOL3_OUT_DIM, NULL, pool3_out);
#endif

	#define IP1_DIM (POOL3_OUT_DIM * POOL3_OUT_DIM * CONV3_OUT_CH)
	#define IP1_OUT 10
	#define IP1_BIAS_LSHIFT (CONV3_OUT_Q+W_FC1_Q-B_FC1_Q)
	#define IP1_OUT_RSHIFT  (CONV3_OUT_Q+W_FC1_Q-FC1_OUT_Q)
#ifndef USE_NNOM
    q7_t W_fc1[IP1_DIM *IP1_OUT] = W_FC1_OPT;
    q7_t b_fc1[IP1_OUT] = B_FC1;
	arm_fully_connected_q7_opt(pool3_out, W_fc1, IP1_DIM, IP1_OUT, IP1_BIAS_LSHIFT, IP1_OUT_RSHIFT, b_fc1,
						  fc1_out, (q15_t*)conv1_out);
	arm_softmax_q7(fc1_out, IP1_OUT, y_out);
    for (int i = 0; i < 10; i++)
    {
        printf("%d: %3d = %3d.%d%%\n", i, y_out[i], y_out[i]*100/128, (y_out[i]*1000/128)%10);
    }
    printf("inference is done!\n");
#endif
	return 0;
}
