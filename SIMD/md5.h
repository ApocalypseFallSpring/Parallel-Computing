#include <iostream>
#include <string>
#include <cstring>
#include <arm_neon.h>
#include <vector>

using namespace std;

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef unsigned int bit32;

// MD5的一系列参数。参数是固定的，其实你不需要看懂这些
#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21

/**
 * @Basic MD5 functions.
 *
 * @param there bit32.
 *
 * @return one bit32.
 */
// 定义了一系列MD5中的具体函数
// 这四个计算函数是需要你进行SIMD并行化的
// 可以看到，FGHI四个函数都涉及一系列位运算，在数据上是对齐的，非常容易实现SIMD的并行化

#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

#define F_SIMD(x, y, z) vorrq_u32(vandq_u32((x), (y)), vandq_u32(vmvnq_u32(x), (z)))
#define G_SIMD(x, y, z) vorrq_u32(vandq_u32((x), (z)), vandq_u32((y), vmvnq_u32(z)))
#define H_SIMD(x, y, z) veorq_u32((x), veorq_u32((y), (z)))
#define I_SIMD(x, y, z) veorq_u32((y), vorrq_u32((x), vmvnq_u32(z)))

#define F_SIMD_2(x, y, z) vorr_u32(vand_u32((x), (y)), vand_u32(vmvn_u32(x), (z)))
#define G_SIMD_2(x, y, z) vorr_u32(vand_u32((x), (z)), vand_u32((y), vmvn_u32(z)))
#define H_SIMD_2(x, y, z) veor_u32((x), veor_u32((y), (z)))
#define I_SIMD_2(x, y, z) veor_u32((y), vorr_u32((x), vmvn_u32(z)))


/**
 * @Rotate Left.
 *
 * @param {num} the raw number.
 *
 * @param {n} rotate left n.
 *
 * @return the number after rotated left.
 */
// 定义了一系列MD5中的具体函数
// 这五个计算函数（ROTATELEFT/FF/GG/HH/II）和之前的FGHI一样，都是需要你进行SIMD并行化的
// 但是你需要注意的是#define的功能及其效果，可以发现这里的FGHI是没有返回值的，为什么呢？你可以查询#define的含义和用法
#define ROTATELEFT(num, n) (((num) << (n)) | ((num) >> (32-(n))))
#define ROTATELEFT_SIMD(num, n) vorrq_u32(vshlq_n_u32((num), (n)), vshrq_n_u32((num), (32 - (n))))
#define ROTATELEFT_SIMD_2(num, n) vorr_u32(vshl_n_u32((num), (n)), vshr_n_u32((num), (32 - (n))))

#define FF(a, b, c, d, x, s, ac) { \
  bit32 t = F((b), (c), (d)); \
  (a) += t + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

#define GG(a, b, c, d, x, s, ac) { \
  (a) += G ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

#define HH(a, b, c, d, x, s, ac) { \
  (a) += H ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

#define II(a, b, c, d, x, s, ac) { \
  (a) += I ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

// #define FF_SIMD(a, b, c, d, x, s, ac) { \
//   (a) = vaddq_u32((a), F_SIMD((b), (c), (d))); \
//   (a) = vaddq_u32((a), (x)); \
//   (a) = vaddq_u32((a), vdupq_n_u32(ac)); \
//   (a) = ROTATELEFT_SIMD((a), (s)); \
//   (a) = vaddq_u32((a), (b)); \
// }

// #define GG_SIMD(a, b, c, d, x, s, ac) { \
//   (a) = vaddq_u32((a), G_SIMD((b), (c), (d))); \
//   (a) = vaddq_u32((a), (x)); \
//   (a) = vaddq_u32((a), vdupq_n_u32(ac)); \
//   (a) = ROTATELEFT_SIMD((a), (s)); \
//   (a) = vaddq_u32((a), (b)); \
// }

// #define HH_SIMD(a, b, c, d, x, s, ac) { \
//   (a) = vaddq_u32((a), H_SIMD((b), (c), (d))); \
//   (a) = vaddq_u32((a), (x)); \
//   (a) = vaddq_u32((a), vdupq_n_u32(ac)); \
//   (a) = ROTATELEFT_SIMD((a), (s)); \
//   (a) = vaddq_u32((a), (b)); \
// }

// #define II_SIMD(a, b, c, d, x, s, ac) { \
//   (a) = vaddq_u32((a), I_SIMD((b), (c), (d))); \
//   (a) = vaddq_u32((a), (x)); \
//   (a) = vaddq_u32((a), vdupq_n_u32(ac)); \
//   (a) = ROTATELEFT_SIMD((a), (s)); \
//   (a) = vaddq_u32((a), (b)); \
// }

// 优化后的SIMD宏，合并加法步骤以减少指令数量
#define FF_SIMD(a, b, c, d, x, s, ac) { \
  uint32x4_t temp = vaddq_u32(vaddq_u32(F_SIMD((b), (c), (d)), (x)), vdupq_n_u32(ac)); \
  (a) = vaddq_u32((a), temp); \
  (a) = ROTATELEFT_SIMD((a), (s)); \
  (a) = vaddq_u32((a), (b)); \
}

#define GG_SIMD(a, b, c, d, x, s, ac) { \
  uint32x4_t temp = vaddq_u32(vaddq_u32(G_SIMD((b), (c), (d)), (x)), vdupq_n_u32(ac)); \
  (a) = vaddq_u32((a), temp); \
  (a) = ROTATELEFT_SIMD((a), (s)); \
  (a) = vaddq_u32((a), (b)); \
}

#define HH_SIMD(a, b, c, d, x, s, ac) { \
  uint32x4_t temp = vaddq_u32(vaddq_u32(H_SIMD((b), (c), (d)), (x)), vdupq_n_u32(ac)); \
  (a) = vaddq_u32((a), temp); \
  (a) = ROTATELEFT_SIMD((a), (s)); \
  (a) = vaddq_u32((a), (b)); \
}

#define II_SIMD(a, b, c, d, x, s, ac) { \
  uint32x4_t temp = vaddq_u32(vaddq_u32(I_SIMD((b), (c), (d)), (x)), vdupq_n_u32(ac)); \
  (a) = vaddq_u32((a), temp); \
  (a) = ROTATELEFT_SIMD((a), (s)); \
  (a) = vaddq_u32((a), (b)); \
}

#define FF_SIMD_2(a, b, c, d, x, s, ac) { \
  uint32x2_t temp = vadd_u32(vadd_u32(F_SIMD_2((b), (c), (d)), (x)), vdup_n_u32(ac)); \
  (a) = vadd_u32((a), temp); \
  (a) = ROTATELEFT_SIMD_2((a), (s)); \
  (a) = vadd_u32((a), (b)); \
}

#define GG_SIMD_2(a, b, c, d, x, s, ac) { \
  uint32x2_t temp = vadd_u32(vadd_u32(G_SIMD_2((b), (c), (d)), (x)), vdup_n_u32(ac)); \
  (a) = vadd_u32((a), temp); \
  (a) = ROTATELEFT_SIMD_2((a), (s)); \
  (a) = vadd_u32((a), (b)); \
}

#define HH_SIMD_2(a, b, c, d, x, s, ac) { \
  uint32x2_t temp = vadd_u32(vadd_u32(H_SIMD_2((b), (c), (d)), (x)), vdup_n_u32(ac)); \
  (a) = vadd_u32((a), temp); \
  (a) = ROTATELEFT_SIMD_2((a), (s)); \
  (a) = vadd_u32((a), (b)); \
}

#define II_SIMD_2(a, b, c, d, x, s, ac) { \
  uint32x2_t temp = vadd_u32(vadd_u32(I_SIMD_2((b), (c), (d)), (x)), vdup_n_u32(ac)); \
  (a) = vadd_u32((a), temp); \
  (a) = ROTATELEFT_SIMD_2((a), (s)); \
  (a) = vadd_u32((a), (b)); \
}


// #define FF_SIMD_8(a, b, c, d, x, s, ac) { \
//   uint32x4_t temp = vaddq_u32(vaddq_u32(F_SIMD((b), (c), (d)), (x)), vdupq_n_u32(ac)); \
//   (a) = vaddq_u32((a), temp); \
//   (a) = ROTATELEFT_SIMD((a), (s)); \
//   (a) = vaddq_u32((a), (b)); \
// }

#define F_SIMD_8(x, y, z) { \
  vorrq_u32(vandq_u32((x).val[0], (y).val[0]), vandq_u32(vmvnq_u32((x).val[0]), (z).val[0])), \
  vorrq_u32(vandq_u32((x).val[1], (y).val[1]), vandq_u32(vmvnq_u32((x).val[1]), (z).val[1]))  \
}

// G(x, y, z) = ((x & z) | (y & ~z))
#define G_SIMD_8(x, y, z) { \
  vorrq_u32(vandq_u32((x).val[0], (z).val[0]), vandq_u32((y).val[0], vmvnq_u32((z).val[0]))), \
  vorrq_u32(vandq_u32((x).val[1], (z).val[1]), vandq_u32((y).val[1], vmvnq_u32((z).val[1])))  \
}

// H(x, y, z) = (x ^ y ^ z)
#define H_SIMD_8(x, y, z){ \
  veorq_u32(veorq_u32((x).val[0], (y).val[0]), (z).val[0]), \
  veorq_u32(veorq_u32((x).val[1], (y).val[1]), (z).val[1])  \
}

// I(x, y, z) = (y ^ (x | ~z))
#define I_SIMD_8(x, y, z) { \
  veorq_u32((y).val[0], vorrq_u32((x).val[0], vmvnq_u32((z).val[0]))), \
  veorq_u32((y).val[1], vorrq_u32((x).val[1], vmvnq_u32((z).val[1])))  \
}

#define FF_SIMD_8(a, b, c, d, x, s, ac) do { \
  /* 计算 F(b, c, d) */ \
  uint32x4x2_t _f_temp = F_SIMD_8((b), (c), (d)); \
  /* 处理 val[0] (前4个元素) */ \
  uint32x4_t _temp0 = vaddq_u32(vaddq_u32(_f_temp.val[0], (x).val[0]), vdupq_n_u32(ac)); \
  (a).val[0] = vaddq_u32((a).val[0], _temp0); \
  (a).val[0] = ROTATELEFT_SIMD((a).val[0], (s)); \
  (a).val[0] = vaddq_u32((a).val[0], (b).val[0]); \
  \
  /* 处理 val[1] (后4个元素) */ \
  uint32x4_t _temp1 = vaddq_u32(vaddq_u32(_f_temp.val[1], (x).val[1]), vdupq_n_u32(ac)); \
  (a).val[1] = vaddq_u32((a).val[1], _temp1); \
  (a).val[1] = ROTATELEFT_SIMD((a).val[1], (s)); \
  (a).val[1] = vaddq_u32((a).val[1], (b).val[1]); \
} while(0)

#define GG_SIMD_8(a, b, c, d, x, s, ac) do { \
  uint32x4x2_t _g_temp = G_SIMD_8((b), (c), (d)); \
  \
  uint32x4_t _temp0 = vaddq_u32(vaddq_u32(_g_temp.val[0], (x).val[0]), vdupq_n_u32(ac)); \
  (a).val[0] = vaddq_u32((a).val[0], _temp0); \
  (a).val[0] = ROTATELEFT_SIMD((a).val[0], (s)); \
  (a).val[0] = vaddq_u32((a).val[0], (b).val[0]); \
  \
  uint32x4_t _temp1 = vaddq_u32(vaddq_u32(_g_temp.val[1], (x).val[1]), vdupq_n_u32(ac)); \
  (a).val[1] = vaddq_u32((a).val[1], _temp1); \
  (a).val[1] = ROTATELEFT_SIMD((a).val[1], (s)); \
  (a).val[1] = vaddq_u32((a).val[1], (b).val[1]); \
} while(0)


#define HH_SIMD_8(a, b, c, d, x, s, ac) do { \
  uint32x4x2_t _h_temp = H_SIMD_8((b), (c), (d)); \
  \
  uint32x4_t _temp0 = vaddq_u32(vaddq_u32(_h_temp.val[0], (x).val[0]), vdupq_n_u32(ac)); \
  (a).val[0] = vaddq_u32((a).val[0], _temp0); \
  (a).val[0] = ROTATELEFT_SIMD((a).val[0], (s)); \
  (a).val[0] = vaddq_u32((a).val[0], (b).val[0]); \
  \
  uint32x4_t _temp1 = vaddq_u32(vaddq_u32(_h_temp.val[1], (x).val[1]), vdupq_n_u32(ac)); \
  (a).val[1] = vaddq_u32((a).val[1], _temp1); \
  (a).val[1] = ROTATELEFT_SIMD((a).val[1], (s)); \
  (a).val[1] = vaddq_u32((a).val[1], (b).val[1]); \
} while(0)

#define II_SIMD_8(a, b, c, d, x, s, ac) do { \
  uint32x4x2_t _i_temp = I_SIMD_8((b), (c), (d)); \
  \
  uint32x4_t _temp0 = vaddq_u32(vaddq_u32(_i_temp.val[0], (x).val[0]), vdupq_n_u32(ac)); \
  (a).val[0] = vaddq_u32((a).val[0], _temp0); \
  (a).val[0] = ROTATELEFT_SIMD((a).val[0], (s)); \
  (a).val[0] = vaddq_u32((a).val[0], (b).val[0]); \
  \
  uint32x4_t _temp1 = vaddq_u32(vaddq_u32(_i_temp.val[1], (x).val[1]), vdupq_n_u32(ac)); \
  (a).val[1] = vaddq_u32((a).val[1], _temp1); \
  (a).val[1] = ROTATELEFT_SIMD((a).val[1], (s)); \
  (a).val[1] = vaddq_u32((a).val[1], (b).val[1]); \
} while(0)
// ---------------------------------------------------------------

void MD5Hash(string input, bit32 *state);

void MD5HashSIMD(const vector<string>& inputs, vector<bit32*>& states);

void MD5HashSIMD2(const vector<string>& inputs, vector<bit32*>& states);

void MD5HashSIMD8(const vector<string>& inputs, vector<bit32*>& states);
