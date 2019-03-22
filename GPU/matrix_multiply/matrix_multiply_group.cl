__kernel void matrix_multiply_group(const int M,
                                const int K,
                                const int N,
                                __global const float *x, 
                                __global const float *y, 
                                __global float *restrict z)
{   
    
    int i = get_global_id(0);
    int j = get_global_id(1);

}
