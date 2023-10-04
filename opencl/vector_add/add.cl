__kernel void add(__global int* a, __global int* b, __global int* c) {
  
  int idx = get_global_id(0);
  c[idx] = a[idx] + b[idx];
  
}
