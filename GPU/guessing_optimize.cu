#include "PCFG.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// CUDA核函数：优化后的口令生成
__global__ void generateGuessesKernel(
    const char* d_guess, 
    int prefix_len,
    const char* d_values,
    const int* d_value_offsets,
    const int* d_value_lengths,
    char* d_output,
    const int* d_output_offsets,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        char* out_ptr = d_output + d_output_offsets[idx];
        
        // 复制密码前缀
        for (int i = 0; i < prefix_len; i++) {
            out_ptr[i] = d_guess[i];
        }
        
        // 复制当前值
        int value_offset = d_value_offsets[idx];
        for (int i = 0; i < d_value_lengths[idx]; i++) {
            out_ptr[prefix_len + i] = d_values[value_offset + i];
        }
    }
}

// 优化后的GPU加速函数
void PriorityQueue::GenerateWithGPU(const std::string& guess, segment* a, int n) {
    // 1. 准备数据 - 优化数据结构减少传输量
    const int prefix_len = guess.length();
    std::vector<int> value_lengths(n);
    std::vector<int> value_offsets(n + 1, 0);
    std::string flat_values;
    
    // 计算值的扁平化存储和偏移量
    for (int i = 0; i < n; ++i) {
        value_lengths[i] = a->ordered_values[i].size();
        value_offsets[i] = flat_values.size();
        flat_values += a->ordered_values[i];
    }
    value_offsets[n] = flat_values.size();  // 结束位置标记
    
    // 计算输出缓冲区大小和偏移
    std::vector<int> output_offsets(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        output_offsets[i + 1] = output_offsets[i] + prefix_len + value_lengths[i];
    }
    const int total_output_size = output_offsets[n];
    
    // 2. 设备内存分配
    char *d_guess = nullptr, *d_values = nullptr, *d_output = nullptr;
    int *d_value_offsets = nullptr, *d_value_lengths = nullptr, *d_output_offsets = nullptr;
    
    cudaMalloc(&d_guess, prefix_len);
    cudaMalloc(&d_values, flat_values.size());
    cudaMalloc(&d_value_offsets, (n + 1) * sizeof(int));
    cudaMalloc(&d_value_lengths, n * sizeof(int));
    cudaMalloc(&d_output, total_output_size);
    cudaMalloc(&d_output_offsets, (n + 1) * sizeof(int));
    
    // 3. 数据传输到GPU
    cudaMemcpy(d_guess, guess.c_str(), prefix_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, flat_values.c_str(), flat_values.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value_offsets, value_offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value_lengths, value_lengths.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_offsets, output_offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    // 4. 配置并启动核函数
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    
    generateGuessesKernel<<<gridSize, blockSize>>>(
        d_guess, prefix_len,
        d_values, d_value_offsets,
        d_value_lengths,
        d_output,
        d_output_offsets,
        n
    );
    
    // 5. 等待核函数执行完成
    cudaDeviceSynchronize();
    
    // 6. 将结果复制回主机
    char* h_output = new char[total_output_size];
    cudaMemcpy(h_output, d_output, total_output_size, cudaMemcpyDeviceToHost);
    
    // 7. 构建最终结果列表
    for (int i = 0; i < n; ++i) {
        guesses.emplace_back(h_output + output_offsets[i], value_lengths[i] + prefix_len);
    }
    
    // 8. 释放资源
    delete[] h_output;
    cudaFree(d_guess);
    cudaFree(d_values);
    cudaFree(d_value_offsets);
    cudaFree(d_value_lengths);
    cudaFree(d_output);
    cudaFree(d_output_offsets);
    
    total_guesses += n;
}

// 修改后的Generate函数
void PriorityQueue::Generate(PT pt) {
    CalProb(pt);

    if (pt.content.size() == 1) {
        segment *a = nullptr;
        if (pt.content[0].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[0])];
        } else if (pt.content[0].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[0])];
        } else if (pt.content[0].type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        // 智能选择执行路径
        const int GPU_THRESHOLD = 10000;
        if (a->ordered_values.size() <= GPU_THRESHOLD) {
            for (int i = 0; i < pt.max_indices[0]; i++) {
                guesses.emplace_back(a->ordered_values[i]);
                total_guesses++;
            }
        } else {
            GenerateWithGPU("", a, pt.max_indices[0]);
        }
    } else {
        string guess_prefix;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (seg_idx == pt.content.size() - 1) break;
            
            if (pt.content[seg_idx].type == 1) {
                guess_prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 2) {
                guess_prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 3) {
                guess_prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx++;
        }

        segment *a = nullptr;
        if (pt.content.back().type == 1) {
            a = &m.letters[m.FindLetter(pt.content.back())];
        } else if (pt.content.back().type == 2) {
            a = &m.digits[m.FindDigit(pt.content.back())];
        } else if (pt.content.back().type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content.back())];
        }
        
        // 智能选择执行路径
        const int GPU_THRESHOLD = 10000;
        if (a->ordered_values.size() <= GPU_THRESHOLD) {
            for (int i = 0; i < pt.max_indices.back(); i++) {
                guesses.emplace_back(guess_prefix + a->ordered_values[i]);
                total_guesses++;
            }
        } else {
            GenerateWithGPU(guess_prefix, a, pt.max_indices.back());
        }
    }
}

// 以下为未修改的原始代码部分
void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}
