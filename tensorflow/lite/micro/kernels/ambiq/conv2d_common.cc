#include <stdint.h>
#include "Include/arm_nnfunctions.h"
#include "Include/arm_nnsupportfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"

arm_cmsis_nn_status arm_convolve_1_x_n_dilated_s8(const cmsis_nn_context *ctx,
                                                  const cmsis_nn_conv_params *conv_params,
                                                  const cmsis_nn_per_channel_quant_params *quant_params,
                                                  const cmsis_nn_dims *input_dims,
                                                  const int8_t *input_data,
                                                  const cmsis_nn_dims *filter_dims,
                                                  const int8_t *filter_data,
                                                  const cmsis_nn_dims *bias_dims,
                                                  const int32_t *bias_data,
                                                  const cmsis_nn_dims *output_dims,
                                                  int8_t *output_data)
{
    // Check: height must be 1
    if (input_dims->h != 1 || filter_dims->h != 1 || output_dims->h != 1)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    // Extract dimensions
    const int32_t batches      = input_dims->n;
    const int32_t input_width  = input_dims->w;
    const int32_t input_ch     = input_dims->c;  // "C_in"

    const int32_t filter_width = filter_dims->w; // "N"
    const int32_t output_width = output_dims->w;
    const int32_t output_ch    = output_dims->c; // "C_out"

    // Convolution parameters
    const int32_t stride_x     = conv_params->stride.w;
    const int32_t pad_x        = conv_params->padding.w;
    const int32_t dilation_x   = conv_params->dilation.w;
    const int32_t input_offset  = conv_params->input_offset;
    const int32_t output_offset = conv_params->output_offset;
    const int32_t act_min       = conv_params->activation.min;
    const int32_t act_max       = conv_params->activation.max;

    // Per-channel quant
    const int32_t *out_mult  = quant_params->multiplier;
    const int32_t *out_shift = quant_params->shift;

    // Scratch buffer for im2row
    int8_t *im2row_buf = (int8_t *)ctx->buf;
    // Check if ctx->size is sufficient (omitted for brevity)

    // Batch loop
    for (int b = 0; b < batches; b++)
    {
        // Base pointers for this batch
        const int8_t *input_data_b  = input_data  + b * (input_width * input_ch);
        int8_t       *output_data_b = output_data + b * (output_width * output_ch);

        // For each output column
        for (int out_x = 0; out_x < output_width; out_x++)
        {
            // Compute origin of input in X dimension
            const int in_x_origin = (out_x * stride_x) - pad_x;

            // 1) Build im2row buffer: (filter_width * input_ch)
            for (int fw = 0; fw < filter_width; fw++)
            {
                const int dilated_x = in_x_origin + fw * dilation_x;
                int8_t *dst = im2row_buf + fw * input_ch;

                if (dilated_x < 0 || dilated_x >= input_width)
                {
                    // Out-of-bounds => fill with zero
                    arm_memset_s8(dst, 0, input_ch);
                }
                else
                {
                    // In-bounds => copy raw input (no offset added yet)
                    const int8_t *src = input_data_b + dilated_x * input_ch;
                    arm_memcpy_s8(dst, src, input_ch);
                }
            }

            // 2) Compute each output channel
            for (int c_out = 0; c_out < output_ch; c_out++)
            {
                // Accumulator (start with bias if available)
                int32_t acc = (bias_data) ? bias_data[c_out] : 0;

                // Filter pointer for this channel
                const int8_t *filter_ptr = filter_data + c_out * (filter_width * input_ch);

                const int total_elems = filter_width * input_ch;
                int i = 0;

                // Vector offset for saturating add
                const int8x16_t offset_vec = vdupq_n_s8((int8_t)input_offset);

                // Process in blocks of 16
                while (i <= (total_elems - 16))
                {
                    // Load 16 input bytes
                    int8x16_t in_vec = vldrbq_s8(&im2row_buf[i]);
                    // Load 16 filter bytes
                    int8x16_t wt_vec = vldrbq_s8(&filter_ptr[i]);

                    // Apply offset with saturating add
                    in_vec = vqaddq_s8(in_vec, offset_vec);

                    // Multiply int8 => int16 partial (lower 8)
                    int16x8_t partial_lo = vmullbq_int_s8(in_vec, wt_vec);
                    // Multiply int8 => int16 partial (upper 8)
                    int16x8_t partial_hi = vmulltq_int_s8(in_vec, wt_vec);

                    // Sum each int16x8_t horizontally into 32-bit
                    int32_t sum_lo = vaddlvq_s16(partial_lo);
                    int32_t sum_hi = vaddlvq_s16(partial_hi);

                    acc += (sum_lo + sum_hi);
                    i += 16;
                }

                // Remainder loop
                for (; i < total_elems; i++)
                {
                    int8_t val = im2row_buf[i];
                    // Saturating add offset in scalar form
                    int16_t tmp = (int16_t)val + (int16_t)input_offset;
                    if (tmp > 127)  tmp = 127;
                    if (tmp < -128) tmp = -128;

                    acc += ((int32_t)tmp) * ((int32_t)filter_ptr[i]);
                }

                // 3) Requantize & store
                acc = mul_shift_quantized(acc, out_mult[c_out], out_shift[c_out]);
                acc += output_offset;
                if (acc < act_min) acc = act_min;
                if (acc > act_max) acc = act_max;

                output_data_b[out_x * output_ch + c_out] = (int8_t)acc;
            }
        }
    }
    return ARM_CMSIS_NN_SUCCESS;
}
