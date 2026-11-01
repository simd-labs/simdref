vfmadd231ps (%rsi,%rax,4),%ymm2,%ymm1
vfmadd231ps %ymm2,%ymm1,%ymm0
vpermd (%rsi),%ymm2,%ymm1
vpermd %ymm2,%ymm1,%ymm0
