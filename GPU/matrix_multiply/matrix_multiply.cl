__kernel void matrix_multiply(const int M,
                                const int K,
                                const int N,
                                __global const float *x, 
                                __global const float *y, 
                                __global float *restrict z)
{    
    int TS = get_local_size(0);
    int i = get_group_id(0) * TS + get_local_id(0);
    int j = get_group_id(1) * TS + get_local_id(1);

    // int i = get_global_id(0);
    // int j = get_global_id(1);


    float res = 0;
    for (int k = 0; k < K; k++){
        res += x[i * K + k] * y[k * N + j];
    };
    z[i * N + j] = res;
}
