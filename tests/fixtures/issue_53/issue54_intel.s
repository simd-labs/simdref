vfmadd231ps ymm1, ymm2, [rsi+rax*4]
vfmadd231ps ymm0, ymm1, ymm2
vpermd ymm1, ymm2, [rsi]
vpermd ymm0, ymm1, ymm2
