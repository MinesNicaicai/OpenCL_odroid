__kernel void videofilter(__global const float *input, 
                        const int input_width,
                        const int input_height,
                        __global const float *ker, 
                        const int ker_width,
                        const int ker_height,
                        __global float *restrict output)
{    
    int index_col = get_global_id(0);
    int index_row = get_global_id(1);
    int left_top_row = index_row - ker_height / 2;
    int left_top_col = index_col - ker_width / 2;
    int right_bot_row = index_row + ker_height/2;
    int right_bot_col = index_col + ker_width/2;
    float res = 0;
    if (left_top_row < 0 || left_top_col < 0 || 
        right_bot_row >= input_height || right_bot_col >= input_width){
        for (int i = 0; i < ker_height; i++){
            for (int j = 0; j < ker_width; j++){
                res += (left_top_row + i < 0 || left_top_col + j < 0 || 
                        right_bot_row >= input_height || right_bot_col >= input_width)
                        ? 0
                        : input[left_top_col + j + (left_top_row + i) * input_width] 
                        * ker[ker_width - 1 - j + (ker_height - 1 - i) * ker_width];
            }
        }
    }
    else{
        for (int i = 0; i < ker_height; i++){
            for (int j = 0; j < ker_width; j++){
                res += input[left_top_col + j + (left_top_row + i) * input_width] 
                    * ker[ker_width - 1 - j + (ker_height - 1 - i) * ker_width];
            }
        }
    }
    output[index_col + index_row * input_width] = res;

    // printf("res = %f\n", res);
}
