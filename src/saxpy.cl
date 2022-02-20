__kernel void saxpy(const global float *x,
                    __global float * y,
                    const float a)
{
  uint gid = get_global_id(0);
  y[gid] = a * x[gid] + y[gid];
}
