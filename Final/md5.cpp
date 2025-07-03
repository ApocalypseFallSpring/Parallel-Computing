#include "md5.h"
#include <iomanip>
#include <assert.h>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

void print_uint32x4x2_t(const char* name, uint32x4x2_t vec) {
    alignas(16) uint32_t temp[8];
    vst1q_u32(&temp[0], vec.val[0]);
    vst1q_u32(&temp[4], vec.val[1]);
    
    printf("%s:\n", name);
    for (int i = 0; i < 8; i++) {
        printf("  [%d] = 0x%08x\n", i, temp[i]);
    }
}

/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组
 * @param input 输入
 * @param[out] n_byte 用于给调用者传递额外的返回值，即最终Byte数组的长度
 * @return Byte消息数组
 */
// Byte *StringProcess(string input, int *n_byte)
// {
// 	// 将输入的字符串转换为Byte为单位的数组
// 	Byte *blocks = (Byte *)input.c_str();
// 	int length = input.length();

// 	// 计算原始消息长度（以比特为单位）
// 	int bitLength = length * 8;//字节到比特->*8

// 	// paddingBits: 原始消息需要的padding长度（以bit为单位）
// 	// 对于给定的消息，将其补齐至length%512==448为止
// 	// 需要注意的是，即便给定的消息满足length%512==448，也需要再pad 512bits
// 	int paddingBits = bitLength % 512;
// 	if (paddingBits > 448)
// 	{
// 		paddingBits = 512 - (paddingBits - 448);
// 	}
// 	else if (paddingBits < 448)
// 	{
// 		paddingBits = 448 - paddingBits;
// 	}
// 	else if (paddingBits == 448)
// 	{
// 		paddingBits = 512;
// 	}

// 	// 原始消息需要的padding长度（以Byte为单位）
// 	int paddingBytes = paddingBits / 8;
// 	// 创建最终的字节数组
// 	// length + paddingBytes + 8:
// 	// 1. length为原始消息的长度（bits）
// 	// 2. paddingBytes为原始消息需要的padding长度（Bytes）
// 	// 3. 在pad到length%512==448之后，需要额外附加64bits的原始消息长度，即8个bytes
// 	int paddedLength = length + paddingBytes + 8;
// 	Byte *paddedMessage = new Byte[paddedLength];

// 	// 复制原始消息
// 	memcpy(paddedMessage, blocks, length);

// 	// 添加填充字节。填充时，第一位为1，后面的所有位均为0。
// 	// 所以第一个byte是0x80
// 	paddedMessage[length] = 0x80;							 // 添加一个0x80字节
// 	memset(paddedMessage + length + 1, 0, paddingBytes - 1); // 填充0字节

// 	// 添加消息长度（64比特，小端格式）
// 	for (int i = 0; i < 8; ++i)
// 	{
// 		// 特别注意此处应当将bitLength转换为uint64_t
// 		// 这里的length是原始消息的长度
// 		paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
// 	}

// 	// 验证长度是否满足要求。此时长度应当是512bit的倍数
// 	int residual = 8 * paddedLength % 512;
// 	// assert(residual == 0);

// 	// 在填充+添加长度之后，消息被分为n_blocks个512bit的部分
// 	//最终被padded之后，会是一个长度为512bit倍数，包含[原始消息,padding(0),原始消息length]
// 	*n_byte = paddedLength;
// 	return paddedMessage;
// }

Byte *StringProcess(const string& input, int *n_byte) {
    const Byte* blocks = reinterpret_cast<const Byte*>(input.data());
    const size_t length = input.length();

    // 计算原始消息长度（以比特为单位）
    const int bitLength = length * 8;
    
    // 计算需要的填充位数
    int paddingBits = bitLength % 512;
    if (paddingBits > 448) {
        paddingBits = 512 - (paddingBits - 448);
    } else if (paddingBits < 448) {
        paddingBits = 448 - paddingBits;
    } else if (paddingBits == 448) {
        paddingBits = 512;
    }

    // 原始消息需要的填充长度（以字节为单位）
    const int paddingBytes = paddingBits / 8;
    // 创建最终的字节数组
    const int paddedLength = length + paddingBytes + 8;
    
    // 使用对齐内存分配 (16字节对齐)
    Byte *paddedMessage = static_cast<Byte*>(aligned_alloc(16, paddedLength));
    if (!paddedMessage) throw std::bad_alloc();

    // 复制原始消息
    memcpy(paddedMessage, blocks, length);

    // 添加填充字节
    paddedMessage[length] = 0x80;
    if (paddingBytes > 1) {
        memset(paddedMessage + length + 1, 0, paddingBytes - 1);
    }

    // 添加消息长度（64比特，小端格式）
    const uint64_t totalBits = static_cast<uint64_t>(length) * 8;
    for (int i = 0; i < 8; ++i) {
        paddedMessage[length + paddingBytes + i] = (totalBits >> (i * 8)) & 0xFF;
    }

    *n_byte = paddedLength;
    return paddedMessage;
}


/**
 * MD5Hash: 将单个输入字符串转换成MD5
 * @param input 输入
 * @param[out] state 用于给调用者传递额外的返回值，即最终的缓冲区，也就是MD5的结果
 * @return Byte消息数组
 */
void MD5Hash(string input, bit32 *state)
{

	Byte *paddedMessage;
	int *messageLength = new int[1];
	for (int i = 0; i < 1; i += 1)
	{
		paddedMessage = StringProcess(input, &messageLength[i]);
		// cout<<messageLength[i]<<endl;
		assert(messageLength[i] == messageLength[0]);
	}
	int n_blocks = messageLength[0] / 64;//原始消息经哈希后，变为n_blocks个512bit的部分

	// bit32* state= new bit32[4];
	// MD5算法的4个寄存器
	// 表示 MD5 算法的 128 位（4 × 32 位）内部状态
	state[0] = 0x67452301;
	state[1] = 0xefcdab89;
	state[2] = 0x98badcfe;
	state[3] = 0x10325476;

	// 逐block地更新state
	for (int i = 0; i < n_blocks; i += 1)
	{	
		bit32 x[16];

		// 下面的处理，在理解上较为复杂
		// 每次迭代处理一个 32 位整数（4 个字节）
		// 最终将 paddedMessage 的 64 字节（512 位）数据转换为 16 个 32 位整数
		// 每次迭代从 paddedMessage 中读取 4 个连续的字节，并将它们组合成一个 32 位整数
		// i * 64：表示当前处理的块（block）在 paddedMessage 中的起始位置（每个块占 64 字节）
		// 4 * i1：表示当前处理的 32 位整数在 paddedMessage 中的起始位置（每个整数占 4 个字节）。
		for (int i1 = 0; i1 < 16; ++i1)
		{
			x[i1] = (paddedMessage[4 * i1 + i * 64]) |	            //读取第 1 个字节
					(paddedMessage[4 * i1 + 1 + i * 64] << 8) |	    //读取第 2 个字节
					(paddedMessage[4 * i1 + 2 + i * 64] << 16) |	//读取第 3 个字节
					(paddedMessage[4 * i1 + 3 + i * 64] << 24);	    //读取第 4 个字节
		}

		bit32 a = state[0], b = state[1], c = state[2], d = state[3];

		// MD5 算法的核心函数
		// 分别对应 MD5 的四轮计算中的四种非线性操作
		// 通过位运算和加法操作对输入数据进行混淆和扩散，从而确保 MD5 哈希的安全性
		// 每轮对 16 个 32 位整数（x[0] 到 x[15]）进行处理

		auto start = system_clock::now();
		/* Round 1 */
		FF(a, b, c, d, x[0], s11, 0xd76aa478);
		FF(d, a, b, c, x[1], s12, 0xe8c7b756);
		FF(c, d, a, b, x[2], s13, 0x242070db);
		FF(b, c, d, a, x[3], s14, 0xc1bdceee);
		FF(a, b, c, d, x[4], s11, 0xf57c0faf);
		FF(d, a, b, c, x[5], s12, 0x4787c62a);
		FF(c, d, a, b, x[6], s13, 0xa8304613);
		FF(b, c, d, a, x[7], s14, 0xfd469501);
		FF(a, b, c, d, x[8], s11, 0x698098d8);
		FF(d, a, b, c, x[9], s12, 0x8b44f7af);
		FF(c, d, a, b, x[10], s13, 0xffff5bb1);
		FF(b, c, d, a, x[11], s14, 0x895cd7be);
		FF(a, b, c, d, x[12], s11, 0x6b901122);
		FF(d, a, b, c, x[13], s12, 0xfd987193);
		FF(c, d, a, b, x[14], s13, 0xa679438e);
		FF(b, c, d, a, x[15], s14, 0x49b40821);
		
		// printf("0x%08x\n", a);
		// printf("0x%08x\n", b);
		// printf("0x%08x\n", c);
		// printf("0x%08x\n", d);

		/* Round 2 */
		GG(a, b, c, d, x[1], s21, 0xf61e2562);
		GG(d, a, b, c, x[6], s22, 0xc040b340);
		GG(c, d, a, b, x[11], s23, 0x265e5a51);
		GG(b, c, d, a, x[0], s24, 0xe9b6c7aa);
		GG(a, b, c, d, x[5], s21, 0xd62f105d);
		GG(d, a, b, c, x[10], s22, 0x2441453);
		GG(c, d, a, b, x[15], s23, 0xd8a1e681);
		GG(b, c, d, a, x[4], s24, 0xe7d3fbc8);
		GG(a, b, c, d, x[9], s21, 0x21e1cde6);
		GG(d, a, b, c, x[14], s22, 0xc33707d6);
		GG(c, d, a, b, x[3], s23, 0xf4d50d87);
		GG(b, c, d, a, x[8], s24, 0x455a14ed);
		GG(a, b, c, d, x[13], s21, 0xa9e3e905);
		GG(d, a, b, c, x[2], s22, 0xfcefa3f8);
		GG(c, d, a, b, x[7], s23, 0x676f02d9);
		GG(b, c, d, a, x[12], s24, 0x8d2a4c8a);

		// printf("0x%08x\n", a);
		// printf("0x%08x\n", b);
		// printf("0x%08x\n", c);
		// printf("0x%08x\n", d);

		/* Round 3 */
		HH(a, b, c, d, x[5], s31, 0xfffa3942);
		HH(d, a, b, c, x[8], s32, 0x8771f681);
		HH(c, d, a, b, x[11], s33, 0x6d9d6122);
		HH(b, c, d, a, x[14], s34, 0xfde5380c);
		HH(a, b, c, d, x[1], s31, 0xa4beea44);
		HH(d, a, b, c, x[4], s32, 0x4bdecfa9);
		HH(c, d, a, b, x[7], s33, 0xf6bb4b60);
		HH(b, c, d, a, x[10], s34, 0xbebfbc70);
		HH(a, b, c, d, x[13], s31, 0x289b7ec6);
		HH(d, a, b, c, x[0], s32, 0xeaa127fa);
		HH(c, d, a, b, x[3], s33, 0xd4ef3085);
		HH(b, c, d, a, x[6], s34, 0x4881d05);
		HH(a, b, c, d, x[9], s31, 0xd9d4d039);
		HH(d, a, b, c, x[12], s32, 0xe6db99e5);
		HH(c, d, a, b, x[15], s33, 0x1fa27cf8);
		HH(b, c, d, a, x[2], s34, 0xc4ac5665);

		// printf("0x%08x\n", a);
		// printf("0x%08x\n", b);
		// printf("0x%08x\n", c);
		// printf("0x%08x\n", d);

		/* Round 4 */
		II(a, b, c, d, x[0], s41, 0xf4292244);
		II(d, a, b, c, x[7], s42, 0x432aff97);
		II(c, d, a, b, x[14], s43, 0xab9423a7);
		II(b, c, d, a, x[5], s44, 0xfc93a039);
		II(a, b, c, d, x[12], s41, 0x655b59c3);
		II(d, a, b, c, x[3], s42, 0x8f0ccc92);
		II(c, d, a, b, x[10], s43, 0xffeff47d);
		II(b, c, d, a, x[1], s44, 0x85845dd1);
		II(a, b, c, d, x[8], s41, 0x6fa87e4f);
		II(d, a, b, c, x[15], s42, 0xfe2ce6e0);
		II(c, d, a, b, x[6], s43, 0xa3014314);
		II(b, c, d, a, x[13], s44, 0x4e0811a1);
		II(a, b, c, d, x[4], s41, 0xf7537e82);
		II(d, a, b, c, x[11], s42, 0xbd3af235);
		II(c, d, a, b, x[2], s43, 0x2ad7d2bb);
		II(b, c, d, a, x[9], s44, 0xeb86d391);
		
		// printf("0x%08x\n", a);
		// printf("0x%08x\n", b);
		// printf("0x%08x\n", c);
		// printf("0x%08x\n", d);

		state[0] += a;
		state[1] += b;
		state[2] += c;
		state[3] += d;

	}

	// 下面的处理，在理解上较为复杂
	for (int i = 0; i < 4; i++)
	{
		uint32_t value = state[i];
		state[i] = ((value & 0xff) << 24) |		 // 将最低字节移到最高位
				   ((value & 0xff00) << 8) |	 // 将次低字节左移
				   ((value & 0xff0000) >> 8) |	 // 将次高字节右移
				   ((value & 0xff000000) >> 24); // 将最高字节移到最低位
	}
	// 输出最终的hash结果
	// for (int i1 = 0; i1 < 4; i1 += 1)
	// {
	// 	cout << std::setw(8) << std::setfill('0') << hex << state[i1];
	// }
	// cout << endl;

	// 释放动态分配的内存
	// 实现SIMD并行算法的时候，也请记得及时回收内存！
	delete[] paddedMessage;
	delete[] messageLength;
}

void MD5HashSIMD(const vector<string>& inputs, vector<bit32*>& states) {
    // 以batch_size = 4为例，输入4个字符串进行并行处理
	size_t batch_size = inputs.size();
	// 预处理inputs
	vector<Byte*> paddedMessages(batch_size);
    vector<int> messageLengths(batch_size);

	// 遍历每个输入字符串，进行并行预处理
    for (size_t i = 0; i < batch_size; ++i) {
        paddedMessages[i] = StringProcess(inputs[i], &messageLengths[i]);
    }
	int n_block = messageLengths[0] / 64;


    // 初始化状态变量
    uint32x4_t va = vdupq_n_u32(0x67452301);
    uint32x4_t vb = vdupq_n_u32(0xefcdab89);
    uint32x4_t vc = vdupq_n_u32(0x98badcfe);
    uint32x4_t vd = vdupq_n_u32(0x10325476);

	// paddedMessages = [Message1, Message2, Message3, Message4]
	// messageLengths = [Length1, Length2, Length3, Length4]
	// n_blocks = [Block1, Block2, Block3, Block4]
	// va = [0x67452301, 0x67452301, 0x67452301, 0x67452301]
	// vb = [0xefcdab89, 0xefcdab89, 0xefcdab89, 0xefcdab89]
	// vc = [0x98badcfe, 0x98badcfe, 0x98badcfe, 0x98badcfe]
	// vd = [0x10325476, 0x10325476, 0x10325476, 0x10325476]
	// 相同index的一列做计算，可以实现SIMD

    // 遍历批量中的每个口令
	// 假设所有的blocks数目都相同，大小为n_block
	// 依次对一组四个pw的每个block同时进行处理
	for(size_t i = 0; i < n_block; ++i) {
		uint32x4_t x[16];	//保存四个pw

		// for (int j = 0; j < 16; ++j) {
		// 	uint32_t temp[4]; // 用于临时存储 4 个口令的第 j 个 32 位整数
		// 	for (int k = 0; k < 4; ++k) {
		// 		// 从每个口令的填充数据中提取第 j 个 32 位整数
		// 		temp[k] = (paddedMessages[k][4 * j + i * 64]) |                // 读取第 1 个字节
		// 				  (paddedMessages[k][4 * j + 1 + i * 64] << 8) |      // 读取第 2 个字节
		// 				  (paddedMessages[k][4 * j + 2 + i * 64] << 16) |     // 读取第 3 个字节
		// 				  (paddedMessages[k][4 * j + 3 + i * 64] << 24);      // 读取第 4 个字节
		// 	}
		// 	// 将 4 个整数加载到 SIMD 向量 x[j]
		// 	x[j] = vld1q_u32(temp);
		// }

		// for (int j = 0; j < 16; ++j) {
		// 	// 创建一个指针数组，指向每个口令的当前块数据
		// 	const uint32_t* ptrs[4];
		// 	for (int k = 0; k < 4; ++k) {
		// 		ptrs[k] = reinterpret_cast<const uint32_t*>(&paddedMessages[k][i * 64 + j * 4]);
		// 	}
		
		// 	// 使用 vld1q_u32 加载 4 个 32 位整数到 SIMD 向量
		// 	uint32x4_t temp_vec = {ptrs[0][0], ptrs[1][0], ptrs[2][0], ptrs[3][0]};
		
		// 	// 将 SIMD 向量存储到 x[j]
		// 	x[j] = temp_vec;
		// }

		for (int j = 0; j < 16; ++j) {
			const uint32_t* base_ptrs[4] = {
				reinterpret_cast<const uint32_t*>(&paddedMessages[0][i * 64]),
				reinterpret_cast<const uint32_t*>(&paddedMessages[1][i * 64]),
				reinterpret_cast<const uint32_t*>(&paddedMessages[2][i * 64]),
				reinterpret_cast<const uint32_t*>(&paddedMessages[3][i * 64])
			};
		
			// 使用 SIMD 加载 4 个 32 位整数
			uint32x4_t temp_vec = {
				base_ptrs[0][j], base_ptrs[1][j], base_ptrs[2][j], base_ptrs[3][j]
			};
		
			// 将 SIMD 向量存储到 x[j]
			x[j] = temp_vec;
		}

		
		uint32x4_t a = va, b = vb, c = vc, d = vd;

		// Round 1
		FF_SIMD(a, b, c, d, x[0], s11, 0xd76aa478);
		FF_SIMD(d, a, b, c, x[1], s12, 0xe8c7b756);
		FF_SIMD(c, d, a, b, x[2], s13, 0x242070db);
		FF_SIMD(b, c, d, a, x[3], s14, 0xc1bdceee);
		FF_SIMD(a, b, c, d, x[4], s11, 0xf57c0faf);
		FF_SIMD(d, a, b, c, x[5], s12, 0x4787c62a);
		FF_SIMD(c, d, a, b, x[6], s13, 0xa8304613);
		FF_SIMD(b, c, d, a, x[7], s14, 0xfd469501);
		FF_SIMD(a, b, c, d, x[8], s11, 0x698098d8);
		FF_SIMD(d, a, b, c, x[9], s12, 0x8b44f7af);
		FF_SIMD(c, d, a, b, x[10], s13, 0xffff5bb1);
		FF_SIMD(b, c, d, a, x[11], s14, 0x895cd7be);
		FF_SIMD(a, b, c, d, x[12], s11, 0x6b901122);
		FF_SIMD(d, a, b, c, x[13], s12, 0xfd987193);
		FF_SIMD(c, d, a, b, x[14], s13, 0xa679438e);
		FF_SIMD(b, c, d, a, x[15], s14, 0x49b40821);

		// Round 2
		GG_SIMD(a, b, c, d, x[1], s21, 0xf61e2562);
		GG_SIMD(d, a, b, c, x[6], s22, 0xc040b340);
		GG_SIMD(c, d, a, b, x[11], s23, 0x265e5a51);
		GG_SIMD(b, c, d, a, x[0], s24, 0xe9b6c7aa);
		GG_SIMD(a, b, c, d, x[5], s21, 0xd62f105d);
		GG_SIMD(d, a, b, c, x[10], s22, 0x2441453);
		GG_SIMD(c, d, a, b, x[15], s23, 0xd8a1e681);
		GG_SIMD(b, c, d, a, x[4], s24, 0xe7d3fbc8);
		GG_SIMD(a, b, c, d, x[9], s21, 0x21e1cde6);
		GG_SIMD(d, a, b, c, x[14], s22, 0xc33707d6);
		GG_SIMD(c, d, a, b, x[3], s23, 0xf4d50d87);
		GG_SIMD(b, c, d, a, x[8], s24, 0x455a14ed);
		GG_SIMD(a, b, c, d, x[13], s21, 0xa9e3e905);
		GG_SIMD(d, a, b, c, x[2], s22, 0xfcefa3f8);
		GG_SIMD(c, d, a, b, x[7], s23, 0x676f02d9);
		GG_SIMD(b, c, d, a, x[12], s24, 0x8d2a4c8a);

		// Round 3
		HH_SIMD(a, b, c, d, x[5], s31, 0xfffa3942);
		HH_SIMD(d, a, b, c, x[8], s32, 0x8771f681);
		HH_SIMD(c, d, a, b, x[11], s33, 0x6d9d6122);
		HH_SIMD(b, c, d, a, x[14], s34, 0xfde5380c);
		HH_SIMD(a, b, c, d, x[1], s31, 0xa4beea44);
		HH_SIMD(d, a, b, c, x[4], s32, 0x4bdecfa9);
		HH_SIMD(c, d, a, b, x[7], s33, 0xf6bb4b60);
		HH_SIMD(b, c, d, a, x[10], s34, 0xbebfbc70);
		HH_SIMD(a, b, c, d, x[13], s31, 0x289b7ec6);
		HH_SIMD(d, a, b, c, x[0], s32, 0xeaa127fa);
		HH_SIMD(c, d, a, b, x[3], s33, 0xd4ef3085);
		HH_SIMD(b, c, d, a, x[6], s34, 0x4881d05);
		HH_SIMD(a, b, c, d, x[9], s31, 0xd9d4d039);
		HH_SIMD(d, a, b, c, x[12], s32, 0xe6db99e5);
		HH_SIMD(c, d, a, b, x[15], s33, 0x1fa27cf8);
		HH_SIMD(b, c, d, a, x[2], s34, 0xc4ac5665);

		// Round 4
		II_SIMD(a, b, c, d, x[0], s41, 0xf4292244);
		II_SIMD(d, a, b, c, x[7], s42, 0x432aff97);
		II_SIMD(c, d, a, b, x[14], s43, 0xab9423a7);
		II_SIMD(b, c, d, a, x[5], s44, 0xfc93a039);
		II_SIMD(a, b, c, d, x[12], s41, 0x655b59c3);
		II_SIMD(d, a, b, c, x[3], s42, 0x8f0ccc92);
		II_SIMD(c, d, a, b, x[10], s43, 0xffeff47d);
		II_SIMD(b, c, d, a, x[1], s44, 0x85845dd1);
		II_SIMD(a, b, c, d, x[8], s41, 0x6fa87e4f);
		II_SIMD(d, a, b, c, x[15], s42, 0xfe2ce6e0);
		II_SIMD(c, d, a, b, x[6], s43, 0xa3014314);
		II_SIMD(b, c, d, a, x[13], s44, 0x4e0811a1);
		II_SIMD(a, b, c, d, x[4], s41, 0xf7537e82);
		II_SIMD(d, a, b, c, x[11], s42, 0xbd3af235);
		II_SIMD(c, d, a, b, x[2], s43, 0x2ad7d2bb);
		II_SIMD(b, c, d, a, x[9], s44, 0xeb86d391);

		// 更新状态
		va = vaddq_u32(va, a);
		vb = vaddq_u32(vb, b);
		vc = vaddq_u32(vc, c);
		vd = vaddq_u32(vd, d);
	}

	// 提取 SIMD 结果到标量
    vst1q_u32(states[0], va); // 将 va 的 4 个 32 位整数存储到 states[0]
	vst1q_u32(states[1], vb); // 将 vb 的 4 个 32 位整数存储到 states[1]
	vst1q_u32(states[2], vc); // 将 vc 的 4 个 32 位整数存储到 states[2]
	vst1q_u32(states[3], vd); // 将 vd 的 4 个 32 位整数存储到 states[3]

	// 下面的处理，在理解上较为复杂
	// for (int i = 0; i < 4; i++)
	// {
	// 	for (int j = 0; j < batch_size; ++j)
	// 	{
	// 		uint32_t value = states[i][j];
	// 		states[i][j] = ((value & 0xff) << 24) |		 // 将最低字节移到最高位
	// 				((value & 0xff00) << 8) |	 // 将次低字节左移
	// 				((value & 0xff0000) >> 8) |	 // 将次高字节右移
	// 				((value & 0xff000000) >> 24); // 将最高字节移到最低位
	// 	}
	// }

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < batch_size; j += 4) {
			uint32x4_t values = vld1q_u32(&states[i][j]);
			values = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(values)));
			vst1q_u32(&states[i][j], values);
		}
	}
	// 释放内存
    for (size_t i = 0; i < batch_size; ++i) {
        delete[] paddedMessages[i];
    }
}

void MD5HashSIMD2(const vector<string>& inputs, vector<bit32*>& states) {
    // 以 batch_size = 2 为例，输入 2 个字符串进行并行处理
    size_t batch_size = inputs.size();

    // 预处理 inputs
    vector<Byte*> paddedMessages(batch_size);
    vector<int> messageLengths(batch_size);

    // 遍历每个输入字符串，进行并行预处理
    for (size_t i = 0; i < batch_size; ++i) {
        paddedMessages[i] = StringProcess(inputs[i], &messageLengths[i]);
    }
    int n_block = messageLengths[0] / 64;

    // 初始化状态变量
    uint32x2_t va = vdup_n_u32(0x67452301);
    uint32x2_t vb = vdup_n_u32(0xefcdab89);
    uint32x2_t vc = vdup_n_u32(0x98badcfe);
    uint32x2_t vd = vdup_n_u32(0x10325476);

    // 遍历批量中的每个块
    for (size_t i = 0; i < n_block; ++i) {
        uint32x2_t x[16]; // 保存两个口令的当前块

        // 加载每个块的数据
        for (int j = 0; j < 16; ++j) {
            const uint32_t* base_ptrs[2] = {
                reinterpret_cast<const uint32_t*>(&paddedMessages[0][i * 64]),
                reinterpret_cast<const uint32_t*>(&paddedMessages[1][i * 64])
            };

            // 使用 SIMD 加载 2 个 32 位整数
            uint32x2_t temp_vec = {
                base_ptrs[0][j], base_ptrs[1][j]
            };

            // 将 SIMD 向量存储到 x[j]
            x[j] = temp_vec;
        }

        uint32x2_t a = va, b = vb, c = vc, d = vd;

        // Round 1
        FF_SIMD_2(a, b, c, d, x[0], s11, 0xd76aa478);
        FF_SIMD_2(d, a, b, c, x[1], s12, 0xe8c7b756);
        FF_SIMD_2(c, d, a, b, x[2], s13, 0x242070db);
        FF_SIMD_2(b, c, d, a, x[3], s14, 0xc1bdceee);
        FF_SIMD_2(a, b, c, d, x[4], s11, 0xf57c0faf);
        FF_SIMD_2(d, a, b, c, x[5], s12, 0x4787c62a);
        FF_SIMD_2(c, d, a, b, x[6], s13, 0xa8304613);
        FF_SIMD_2(b, c, d, a, x[7], s14, 0xfd469501);
        FF_SIMD_2(a, b, c, d, x[8], s11, 0x698098d8);
        FF_SIMD_2(d, a, b, c, x[9], s12, 0x8b44f7af);
        FF_SIMD_2(c, d, a, b, x[10], s13, 0xffff5bb1);
        FF_SIMD_2(b, c, d, a, x[11], s14, 0x895cd7be);
        FF_SIMD_2(a, b, c, d, x[12], s11, 0x6b901122);
        FF_SIMD_2(d, a, b, c, x[13], s12, 0xfd987193);
        FF_SIMD_2(c, d, a, b, x[14], s13, 0xa679438e);
        FF_SIMD_2(b, c, d, a, x[15], s14, 0x49b40821);

        // Round 2
        GG_SIMD_2(a, b, c, d, x[1], s21, 0xf61e2562);
        GG_SIMD_2(d, a, b, c, x[6], s22, 0xc040b340);
        GG_SIMD_2(c, d, a, b, x[11], s23, 0x265e5a51);
        GG_SIMD_2(b, c, d, a, x[0], s24, 0xe9b6c7aa);
        GG_SIMD_2(a, b, c, d, x[5], s21, 0xd62f105d);
        GG_SIMD_2(d, a, b, c, x[10], s22, 0x2441453);
        GG_SIMD_2(c, d, a, b, x[15], s23, 0xd8a1e681);
        GG_SIMD_2(b, c, d, a, x[4], s24, 0xe7d3fbc8);
        GG_SIMD_2(a, b, c, d, x[9], s21, 0x21e1cde6);
        GG_SIMD_2(d, a, b, c, x[14], s22, 0xc33707d6);
        GG_SIMD_2(c, d, a, b, x[3], s23, 0xf4d50d87);
        GG_SIMD_2(b, c, d, a, x[8], s24, 0x455a14ed);
        GG_SIMD_2(a, b, c, d, x[13], s21, 0xa9e3e905);
        GG_SIMD_2(d, a, b, c, x[2], s22, 0xfcefa3f8);
        GG_SIMD_2(c, d, a, b, x[7], s23, 0x676f02d9);
        GG_SIMD_2(b, c, d, a, x[12], s24, 0x8d2a4c8a);

        // Round 3
        HH_SIMD_2(a, b, c, d, x[5], s31, 0xfffa3942);
        HH_SIMD_2(d, a, b, c, x[8], s32, 0x8771f681);
        HH_SIMD_2(c, d, a, b, x[11], s33, 0x6d9d6122);
        HH_SIMD_2(b, c, d, a, x[14], s34, 0xfde5380c);
        HH_SIMD_2(a, b, c, d, x[1], s31, 0xa4beea44);
        HH_SIMD_2(d, a, b, c, x[4], s32, 0x4bdecfa9);
        HH_SIMD_2(c, d, a, b, x[7], s33, 0xf6bb4b60);
        HH_SIMD_2(b, c, d, a, x[10], s34, 0xbebfbc70);
        HH_SIMD_2(a, b, c, d, x[13], s31, 0x289b7ec6);
        HH_SIMD_2(d, a, b, c, x[0], s32, 0xeaa127fa);
        HH_SIMD_2(c, d, a, b, x[3], s33, 0xd4ef3085);
        HH_SIMD_2(b, c, d, a, x[6], s34, 0x4881d05);
        HH_SIMD_2(a, b, c, d, x[9], s31, 0xd9d4d039);
        HH_SIMD_2(d, a, b, c, x[12], s32, 0xe6db99e5);
        HH_SIMD_2(c, d, a, b, x[15], s33, 0x1fa27cf8);
        HH_SIMD_2(b, c, d, a, x[2], s34, 0xc4ac5665);

        // Round 4
        II_SIMD_2(a, b, c, d, x[0], s41, 0xf4292244);
        II_SIMD_2(d, a, b, c, x[7], s42, 0x432aff97);
        II_SIMD_2(c, d, a, b, x[14], s43, 0xab9423a7);
        II_SIMD_2(b, c, d, a, x[5], s44, 0xfc93a039);
        II_SIMD_2(a, b, c, d, x[12], s41, 0x655b59c3);
        II_SIMD_2(d, a, b, c, x[3], s42, 0x8f0ccc92);
        II_SIMD_2(c, d, a, b, x[10], s43, 0xffeff47d);
        II_SIMD_2(b, c, d, a, x[1], s44, 0x85845dd1);
        II_SIMD_2(a, b, c, d, x[8], s41, 0x6fa87e4f);
        II_SIMD_2(d, a, b, c, x[15], s42, 0xfe2ce6e0);
        II_SIMD_2(c, d, a, b, x[6], s43, 0xa3014314);
        II_SIMD_2(b, c, d, a, x[13], s44, 0x4e0811a1);
        II_SIMD_2(a, b, c, d, x[4], s41, 0xf7537e82);
        II_SIMD_2(d, a, b, c, x[11], s42, 0xbd3af235);
        II_SIMD_2(c, d, a, b, x[2], s43, 0x2ad7d2bb);
        II_SIMD_2(b, c, d, a, x[9], s44, 0xeb86d391);

        // 更新状态
        va = vadd_u32(va, a);
        vb = vadd_u32(vb, b);
        vc = vadd_u32(vc, c);
        vd = vadd_u32(vd, d);
    }

    // 提取 SIMD 结果到标量
    vst1_u32(states[0], va); // 将 va 的 2 个 32 位整数存储到 states[0]
    vst1_u32(states[1], vb); // 将 vb 的 2 个 32 位整数存储到 states[1]
    vst1_u32(states[2], vc); // 将 vc 的 2 个 32 位整数存储到 states[2]
    vst1_u32(states[3], vd); // 将 vd 的 2 个 32 位整数存储到 states[3]

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < batch_size; j += 2) {
			// 加载 2 个 32 位整数
			uint32x2_t values = vld1_u32(&states[i][j]);
			// 字节顺序翻转
			values = vreinterpret_u32_u8(vrev32_u8(vreinterpret_u8_u32(values)));
			// 存储翻转后的结果
			vst1_u32(&states[i][j], values);
		}
	}

    // 释放内存
    for (size_t i = 0; i < batch_size; ++i) {
        delete[] paddedMessages[i];
    }
}


void MD5HashSIMD8(const vector<string>& inputs, vector<bit32*>& states) {
    const size_t batch_size = inputs.size();
    if (batch_size != 8) {
        throw std::invalid_argument("MD5HashSIMD8 requires exactly 8 inputs");
    }

    vector<Byte*> paddedMessages(batch_size);
    vector<int> messageLengths(batch_size);

    // 并行预处理 - 使用OpenMP加速
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size; ++i) {
        paddedMessages[i] = StringProcess(inputs[i], &messageLengths[i]);
    }

    const int n_block = messageLengths[0] / 64;

    // 初始化状态变量
    uint32x4x2_t va = {vdupq_n_u32(0x67452301), vdupq_n_u32(0x67452301)};
    uint32x4x2_t vb = {vdupq_n_u32(0xefcdab89), vdupq_n_u32(0xefcdab89)};
    uint32x4x2_t vc = {vdupq_n_u32(0x98badcfe), vdupq_n_u32(0x98badcfe)};
    uint32x4x2_t vd = {vdupq_n_u32(0x10325476), vdupq_n_u32(0x10325476)};

    // 处理每个块
    for (size_t i = 0; i < n_block; ++i) {
        uint32x4x2_t x[16];
        
        // 预取下个块的数据
        if (i + 1 < n_block) {
            for (size_t k = 0; k < batch_size; ++k) {
                __builtin_prefetch(paddedMessages[k] + (i + 1) * 64);
            }
        }

        // 高效加载当前块的数据
        for (int j = 0; j < 16; ++j) {
            // 低4组
            uint32x4_t temp_vec_low = {
                *reinterpret_cast<const uint32_t*>(&paddedMessages[0][i * 64 + j * 4]),
                *reinterpret_cast<const uint32_t*>(&paddedMessages[1][i * 64 + j * 4]),
                *reinterpret_cast<const uint32_t*>(&paddedMessages[2][i * 64 + j * 4]),
                *reinterpret_cast<const uint32_t*>(&paddedMessages[3][i * 64 + j * 4])
            };
            
            // 高4组
            uint32x4_t temp_vec_high = {
                *reinterpret_cast<const uint32_t*>(&paddedMessages[4][i * 64 + j * 4]),
                *reinterpret_cast<const uint32_t*>(&paddedMessages[5][i * 64 + j * 4]),
                *reinterpret_cast<const uint32_t*>(&paddedMessages[6][i * 64 + j * 4]),
                *reinterpret_cast<const uint32_t*>(&paddedMessages[7][i * 64 + j * 4])
            };
            
            x[j] = {temp_vec_low, temp_vec_high};
        }

        uint32x4x2_t a = va, b = vb, c = vc, d = vd;

        // Round 1
        FF_SIMD_8(a, b, c, d, x[0], s11, 0xd76aa478);
        FF_SIMD_8(d, a, b, c, x[1], s12, 0xe8c7b756);
        FF_SIMD_8(c, d, a, b, x[2], s13, 0x242070db);
        FF_SIMD_8(b, c, d, a, x[3], s14, 0xc1bdceee);
        FF_SIMD_8(a, b, c, d, x[4], s11, 0xf57c0faf);
        FF_SIMD_8(d, a, b, c, x[5], s12, 0x4787c62a);
        FF_SIMD_8(c, d, a, b, x[6], s13, 0xa8304613);
        FF_SIMD_8(b, c, d, a, x[7], s14, 0xfd469501);
        FF_SIMD_8(a, b, c, d, x[8], s11, 0x698098d8);
        FF_SIMD_8(d, a, b, c, x[9], s12, 0x8b44f7af);
        FF_SIMD_8(c, d, a, b, x[10], s13, 0xffff5bb1);
        FF_SIMD_8(b, c, d, a, x[11], s14, 0x895cd7be);
        FF_SIMD_8(a, b, c, d, x[12], s11, 0x6b901122);
        FF_SIMD_8(d, a, b, c, x[13], s12, 0xfd987193);
        FF_SIMD_8(c, d, a, b, x[14], s13, 0xa679438e);
        FF_SIMD_8(b, c, d, a, x[15], s14, 0x49b40821);

        // Round 2
        GG_SIMD_8(a, b, c, d, x[1], s21, 0xf61e2562);
        GG_SIMD_8(d, a, b, c, x[6], s22, 0xc040b340);
        GG_SIMD_8(c, d, a, b, x[11], s23, 0x265e5a51);
        GG_SIMD_8(b, c, d, a, x[0], s24, 0xe9b6c7aa);
        GG_SIMD_8(a, b, c, d, x[5], s21, 0xd62f105d);
        GG_SIMD_8(d, a, b, c, x[10], s22, 0x2441453);
        GG_SIMD_8(c, d, a, b, x[15], s23, 0xd8a1e681);
        GG_SIMD_8(b, c, d, a, x[4], s24, 0xe7d3fbc8);
        GG_SIMD_8(a, b, c, d, x[9], s21, 0x21e1cde6);
        GG_SIMD_8(d, a, b, c, x[14], s22, 0xc33707d6);
        GG_SIMD_8(c, d, a, b, x[3], s23, 0xf4d50d87);
        GG_SIMD_8(b, c, d, a, x[8], s24, 0x455a14ed);
        GG_SIMD_8(a, b, c, d, x[13], s21, 0xa9e3e905);
        GG_SIMD_8(d, a, b, c, x[2], s22, 0xfcefa3f8);
        GG_SIMD_8(c, d, a, b, x[7], s23, 0x676f02d9);
        GG_SIMD_8(b, c, d, a, x[12], s24, 0x8d2a4c8a);

        // Round 3
        HH_SIMD_8(a, b, c, d, x[5], s31, 0xfffa3942);
        HH_SIMD_8(d, a, b, c, x[8], s32, 0x8771f681);
        HH_SIMD_8(c, d, a, b, x[11], s33, 0x6d9d6122);
        HH_SIMD_8(b, c, d, a, x[14], s34, 0xfde5380c);
        HH_SIMD_8(a, b, c, d, x[1], s31, 0xa4beea44);
        HH_SIMD_8(d, a, b, c, x[4], s32, 0x4bdecfa9);
        HH_SIMD_8(c, d, a, b, x[7], s33, 0xf6bb4b60);
        HH_SIMD_8(b, c, d, a, x[10], s34, 0xbebfbc70);
        HH_SIMD_8(a, b, c, d, x[13], s31, 0x289b7ec6);
        HH_SIMD_8(d, a, b, c, x[0], s32, 0xeaa127fa);
        HH_SIMD_8(c, d, a, b, x[3], s33, 0xd4ef3085);
        HH_SIMD_8(b, c, d, a, x[6], s34, 0x4881d05);
        HH_SIMD_8(a, b, c, d, x[9], s31, 0xd9d4d039);
        HH_SIMD_8(d, a, b, c, x[12], s32, 0xe6db99e5);
        HH_SIMD_8(c, d, a, b, x[15], s33, 0x1fa27cf8);
        HH_SIMD_8(b, c, d, a, x[2], s34, 0xc4ac5665);

        // Round 4
        II_SIMD_8(a, b, c, d, x[0], s41, 0xf4292244);
        II_SIMD_8(d, a, b, c, x[7], s42, 0x432aff97);
        II_SIMD_8(c, d, a, b, x[14], s43, 0xab9423a7);
        II_SIMD_8(b, c, d, a, x[5], s44, 0xfc93a039);
        II_SIMD_8(a, b, c, d, x[12], s41, 0x655b59c3);
        II_SIMD_8(d, a, b, c, x[3], s42, 0x8f0ccc92);
        II_SIMD_8(c, d, a, b, x[10], s43, 0xffeff47d);
        II_SIMD_8(b, c, d, a, x[1], s44, 0x85845dd1);
        II_SIMD_8(a, b, c, d, x[8], s41, 0x6fa87e4f);
        II_SIMD_8(d, a, b, c, x[15], s42, 0xfe2ce6e0);
        II_SIMD_8(c, d, a, b, x[6], s43, 0xa3014314);
        II_SIMD_8(b, c, d, a, x[13], s44, 0x4e0811a1);
        II_SIMD_8(a, b, c, d, x[4], s41, 0xf7537e82);
        II_SIMD_8(d, a, b, c, x[11], s42, 0xbd3af235);
        II_SIMD_8(c, d, a, b, x[2], s43, 0x2ad7d2bb);
        II_SIMD_8(b, c, d, a, x[9], s44, 0xeb86d391);

        // 更新状态
        va.val[0] = vaddq_u32(va.val[0], a.val[0]);
        va.val[1] = vaddq_u32(va.val[1], a.val[1]);
        vb.val[0] = vaddq_u32(vb.val[0], b.val[0]);
        vb.val[1] = vaddq_u32(vb.val[1], b.val[1]);
        vc.val[0] = vaddq_u32(vc.val[0], c.val[0]);
        vc.val[1] = vaddq_u32(vc.val[1], c.val[1]);
        vd.val[0] = vaddq_u32(vd.val[0], d.val[0]);
        vd.val[1] = vaddq_u32(vd.val[1], d.val[1]);
    }

    // 存储结果
    auto store_vector = [](uint32x4x2_t vec, bit32* array) {
        vst1q_u32(reinterpret_cast<uint32_t*>(&array[0]), vec.val[0]);
        vst1q_u32(reinterpret_cast<uint32_t*>(&array[4]), vec.val[1]);
    };

    store_vector(va, states[0]);
    store_vector(vb, states[1]);
    store_vector(vc, states[2]);
    store_vector(vd, states[3]);

    // 字节序转换
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < batch_size; j += 4) {
            uint32x4_t values = vld1q_u32(&states[i][j]);
            values = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(values)));
            vst1q_u32(&states[i][j], values);
        }
    }

    // 释放内存
    for (size_t i = 0; i < batch_size; ++i) {
        free(paddedMessages[i]);
    }
}

void MD5HashSIMD16(const vector<string>& inputs, vector<bit32*>& states) {
    size_t batch_size = inputs.size();
    vector<Byte*> paddedMessages(batch_size);
    vector<int> messageLengths(batch_size);

    // 预处理每个输入字符串
    for (size_t i = 0; i < batch_size; ++i) {
        paddedMessages[i] = StringProcess(inputs[i], &messageLengths[i]);
    }
    int n_block = messageLengths[0] / 64;

    // 初始化状态变量
    uint32x4x4_t va, vb, vc, vd;
    for (int i = 0; i < 4; ++i) {
        va.val[i] = vdupq_n_u32(0x67452301);
        vb.val[i] = vdupq_n_u32(0xefcdab89);
        vc.val[i] = vdupq_n_u32(0x98badcfe);
        vd.val[i] = vdupq_n_u32(0x10325476);
    }

    for (size_t i = 0; i < n_block; ++i) {
        uint32x4x4_t x[16];
        // 加载16口令的块数据到 x[0..15]
        for (int j = 0; j < 16; ++j) {
            // 每4个为一组，分别填充到val[0]~val[3]
            for (int k = 0; k < 4; ++k) {
                uint32_t temp[4];
                for (int l = 0; l < 4; ++l) {
                    int idx = k * 4 + l; // 0~15
                    temp[l] = reinterpret_cast<const uint32_t*>(&paddedMessages[idx][i * 64])[j];
                }
                x[j].val[k] = vld1q_u32(temp);
            }
        }

        uint32x4x4_t a = va, b = vb, c = vc, d = vd;

        // Round 1
        FF_SIMD_16(a, b, c, d, x[0], s11, 0xd76aa478);
        FF_SIMD_16(d, a, b, c, x[1], s12, 0xe8c7b756);
        FF_SIMD_16(c, d, a, b, x[2], s13, 0x242070db);
        FF_SIMD_16(b, c, d, a, x[3], s14, 0xc1bdceee);
        FF_SIMD_16(a, b, c, d, x[4], s11, 0xf57c0faf);
        FF_SIMD_16(d, a, b, c, x[5], s12, 0x4787c62a);
        FF_SIMD_16(c, d, a, b, x[6], s13, 0xa8304613);
        FF_SIMD_16(b, c, d, a, x[7], s14, 0xfd469501);
        FF_SIMD_16(a, b, c, d, x[8], s11, 0x698098d8);
        FF_SIMD_16(d, a, b, c, x[9], s12, 0x8b44f7af);
        FF_SIMD_16(c, d, a, b, x[10], s13, 0xffff5bb1);
        FF_SIMD_16(b, c, d, a, x[11], s14, 0x895cd7be);
        FF_SIMD_16(a, b, c, d, x[12], s11, 0x6b901122);
        FF_SIMD_16(d, a, b, c, x[13], s12, 0xfd987193);
        FF_SIMD_16(c, d, a, b, x[14], s13, 0xa679438e);
        FF_SIMD_16(b, c, d, a, x[15], s14, 0x49b40821);

        // Round 2
        GG_SIMD_16(a, b, c, d, x[1], s21, 0xf61e2562);
        GG_SIMD_16(d, a, b, c, x[6], s22, 0xc040b340);
        GG_SIMD_16(c, d, a, b, x[11], s23, 0x265e5a51);
        GG_SIMD_16(b, c, d, a, x[0], s24, 0xe9b6c7aa);
        GG_SIMD_16(a, b, c, d, x[5], s21, 0xd62f105d);
        GG_SIMD_16(d, a, b, c, x[10], s22, 0x2441453);
        GG_SIMD_16(c, d, a, b, x[15], s23, 0xd8a1e681);
        GG_SIMD_16(b, c, d, a, x[4], s24, 0xe7d3fbc8);
        GG_SIMD_16(a, b, c, d, x[9], s21, 0x21e1cde6);
        GG_SIMD_16(d, a, b, c, x[14], s22, 0xc33707d6);
        GG_SIMD_16(c, d, a, b, x[3], s23, 0xf4d50d87);
        GG_SIMD_16(b, c, d, a, x[8], s24, 0x455a14ed);
        GG_SIMD_16(a, b, c, d, x[13], s21, 0xa9e3e905);
        GG_SIMD_16(d, a, b, c, x[2], s22, 0xfcefa3f8);
        GG_SIMD_16(c, d, a, b, x[7], s23, 0x676f02d9);
        GG_SIMD_16(b, c, d, a, x[12], s24, 0x8d2a4c8a);

        // Round 3
        HH_SIMD_16(a, b, c, d, x[5], s31, 0xfffa3942);
        HH_SIMD_16(d, a, b, c, x[8], s32, 0x8771f681);
        HH_SIMD_16(c, d, a, b, x[11], s33, 0x6d9d6122);
        HH_SIMD_16(b, c, d, a, x[14], s34, 0xfde5380c);
        HH_SIMD_16(a, b, c, d, x[1], s31, 0xa4beea44);
        HH_SIMD_16(d, a, b, c, x[4], s32, 0x4bdecfa9);
        HH_SIMD_16(c, d, a, b, x[7], s33, 0xf6bb4b60);
        HH_SIMD_16(b, c, d, a, x[10], s34, 0xbebfbc70);
        HH_SIMD_16(a, b, c, d, x[13], s31, 0x289b7ec6);
        HH_SIMD_16(d, a, b, c, x[0], s32, 0xeaa127fa);
        HH_SIMD_16(c, d, a, b, x[3], s33, 0xd4ef3085);
        HH_SIMD_16(b, c, d, a, x[6], s34, 0x4881d05);
        HH_SIMD_16(a, b, c, d, x[9], s31, 0xd9d4d039);
        HH_SIMD_16(d, a, b, c, x[12], s32, 0xe6db99e5);
        HH_SIMD_16(c, d, a, b, x[15], s33, 0x1fa27cf8);
        HH_SIMD_16(b, c, d, a, x[2], s34, 0xc4ac5665);

        // Round 4
        II_SIMD_16(a, b, c, d, x[0], s41, 0xf4292244);
        II_SIMD_16(d, a, b, c, x[7], s42, 0x432aff97);
        II_SIMD_16(c, d, a, b, x[14], s43, 0xab9423a7);
        II_SIMD_16(b, c, d, a, x[5], s44, 0xfc93a039);
        II_SIMD_16(a, b, c, d, x[12], s41, 0x655b59c3);
        II_SIMD_16(d, a, b, c, x[3], s42, 0x8f0ccc92);
        II_SIMD_16(c, d, a, b, x[10], s43, 0xffeff47d);
        II_SIMD_16(b, c, d, a, x[1], s44, 0x85845dd1);
        II_SIMD_16(a, b, c, d, x[8], s41, 0x6fa87e4f);
        II_SIMD_16(d, a, b, c, x[15], s42, 0xfe2ce6e0);
        II_SIMD_16(c, d, a, b, x[6], s43, 0xa3014314);
        II_SIMD_16(b, c, d, a, x[13], s44, 0x4e0811a1);
        II_SIMD_16(a, b, c, d, x[4], s41, 0xf7537e82);
        II_SIMD_16(d, a, b, c, x[11], s42, 0xbd3af235);
        II_SIMD_16(c, d, a, b, x[2], s43, 0x2ad7d2bb);
        II_SIMD_16(b, c, d, a, x[9], s44, 0xeb86d391);

        // 更新状态
        for (int k = 0; k < 4; ++k) {
            va.val[k] = vaddq_u32(va.val[k], a.val[k]);
            vb.val[k] = vaddq_u32(vb.val[k], b.val[k]);
            vc.val[k] = vaddq_u32(vc.val[k], c.val[k]);
            vd.val[k] = vaddq_u32(vd.val[k], d.val[k]);
        }
    }

    // 存储结果到states
    auto store_vector = [](uint32x4x4_t vec, bit32* array) {
        for (int k = 0; k < 4; ++k) {
            vst1q_u32(reinterpret_cast<uint32_t*>(&array[k * 4]), vec.val[k]);
        }
    };
    store_vector(va, states[0]);
    store_vector(vb, states[1]);
    store_vector(vc, states[2]);
    store_vector(vd, states[3]);

    // 字节序翻转
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < batch_size; j += 4) {
            uint32x4_t values = vld1q_u32(&states[i][j]);
            values = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(values)));
            vst1q_u32(&states[i][j], values);
        }
    }

    // 释放内存
    for (size_t i = 0; i < batch_size; ++i) {
        delete[] paddedMessages[i];
    }
}
