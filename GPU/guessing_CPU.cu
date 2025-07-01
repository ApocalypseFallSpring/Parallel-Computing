#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <algorithm>
#include <thread>

using namespace std;

// 核函数：生成单个PT的猜测口令
__global__ void generateGuessesKernel(
    const char* d_guess, int guess_len,
    const char* d_values, const int* d_value_lengths, int n,
    char* d_output, const int* d_output_offsets
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int out_offset = d_output_offsets[idx];
        char* out_ptr = d_output + out_offset;
        
        // 复制前缀
        for (int i = 0; i < guess_len; i++) {
            out_ptr[i] = d_guess[i];
        }
        
        // 复制当前值
        const char* val_ptr = d_values + idx * 32;  // 假设最大长度为32
        int val_len = d_value_lengths[idx];
        for (int i = 0; i < val_len; i++) {
            out_ptr[guess_len + i] = val_ptr[i];
        }
    }
}

// 辅助函数：使用GPU生成单个PT的猜测
void PriorityQueue::GenerateWithGPU(const string& guess, segment* a, int n) {
    const int GPU_THRESHOLD = 10000;
    if (n <= GPU_THRESHOLD) {
        for (int i = 0; i < n; i++) {
            guesses.emplace_back(guess + a->ordered_values[i]);
        }
        total_guesses += n;
        return;
    }

    // 准备数据
    vector<string>& values = a->ordered_values;
    vector<int> value_lengths(n);
    vector<int> output_offsets(n + 1, 0);
    int total_output_chars = 0;

    for (int i = 0; i < n; i++) {
        value_lengths[i] = values[i].length();
        output_offsets[i + 1] = output_offsets[i] + guess.length() + value_lengths[i];
    }
    total_output_chars = output_offsets[n];

    // 扁平化values
    string flat_values;
    for (const auto& v : values) {
        flat_values += v;
    }

    // 分配设备内存
    char *d_guess, *d_values, *d_output;
    int *d_value_lengths, *d_output_offsets;

    cudaMalloc(&d_guess, guess.size());
    cudaMalloc(&d_values, flat_values.size());
    cudaMalloc(&d_value_lengths, n * sizeof(int));
    cudaMalloc(&d_output_offsets, (n + 1) * sizeof(int));
    cudaMalloc(&d_output, total_output_chars);

    // 拷贝数据到设备
    cudaMemcpy(d_guess, guess.c_str(), guess.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, flat_values.c_str(), flat_values.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value_lengths, value_lengths.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_offsets, output_offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // 启动核函数
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    generateGuessesKernel<<<gridSize, blockSize>>>(
        d_guess, guess.length(),
        d_values, d_value_lengths, n,
        d_output, d_output_offsets
    );

    // 同步设备
    cudaDeviceSynchronize();

    // 拷贝结果回主机
    char* output_chars = new char[total_output_chars];
    cudaMemcpy(output_chars, d_output, total_output_chars, cudaMemcpyDeviceToHost);

    // 构建结果字符串
    for (int i = 0; i < n; i++) {
        guesses.emplace_back(output_chars + output_offsets[i], output_offsets[i + 1] - output_offsets[i]);
    }

    // 清理资源
    delete[] output_chars;
    cudaFree(d_guess);
    cudaFree(d_values);
    cudaFree(d_value_lengths);
    cudaFree(d_output_offsets);
    cudaFree(d_output);

    total_guesses += n;
}

// 批量生成PT的猜测
void PriorityQueue::GenerateBatchWithGPU(const vector<PT>& batch) {
    // 准备批量数据
    vector<string> prefixes;
    vector<segment*> segments;
    vector<int> counts;
    vector<int> total_offsets = {0};  // 累积偏移量

    // 收集每个PT的数据
    for (const auto& pt : batch) {
        // 获取前缀
        string prefix = "";
        if (pt.content.size() > 1) {
            int seg_idx = 0;
            for (int idx : pt.curr_indices) {
                if (pt.content[seg_idx].type == 1) {
                    prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
                } else if (pt.content[seg_idx].type == 2) {
                    prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
                } else if (pt.content[seg_idx].type == 3) {
                    prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
                }
                seg_idx++;
                if (seg_idx == pt.content.size() - 1) break;
            }
        }
        prefixes.push_back(prefix);

        // 获取最后一个segment
        segment* seg_ptr = nullptr;
        if (pt.content.back().type == 1) {
            seg_ptr = &m.letters[m.FindLetter(pt.content.back())];
        } else if (pt.content.back().type == 2) {
            seg_ptr = &m.digits[m.FindDigit(pt.content.back())];
        } else if (pt.content.back().type == 3) {
            seg_ptr = &m.symbols[m.FindSymbol(pt.content.back())];
        }
        segments.push_back(seg_ptr);
        
        // 记录每个PT的猜测数量
        int count = pt.max_indices.back();
        counts.push_back(count);
        total_offsets.push_back(total_offsets.back() + count);
    }

    // 为每个PT生成猜测
    for (size_t i = 0; i < batch.size(); i++) {
        GenerateWithGPU(prefixes[i], segments[i], counts[i]);
    }
}

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

// // 批量处理PT
// void PriorityQueue::PopNextBatch(int batch_size) {
//     vector<PT> batch;
    
//     // 从优先队列中取出batch_size个PT
//     for (int i = 0; i < batch_size && !priority.empty(); i++) {
//         batch.push_back(priority.front());
//         priority.erase(priority.begin());
//     }

//     // 为整个batch生成猜测
//     GenerateBatchWithGPU(batch);

//     // 为每个PT生成新PT并收集
//     vector<PT> new_pts_all;
//     for (auto& pt : batch) {
//         vector<PT> new_pts = pt.NewPTs();
//         new_pts_all.insert(new_pts_all.end(), new_pts.begin(), new_pts.end());
//     }

//     // 将新PT插入优先队列
//     for (PT& new_pt : new_pts_all) {
//         CalProb(new_pt);
//         auto it = priority.begin();
//         while (it != priority.end() && it->prob > new_pt.prob) {
//             ++it;
//         }
//         priority.insert(it, new_pt);
//     }

//     total_guesses += batch.size();
// }

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
        GenerateWithGPU("", a, pt.max_indices[0]);
    } else {
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1) {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 2) {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 3) {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx++;
            if (seg_idx == pt.content.size() - 1) break;
        }

        segment *a = nullptr;
        if (pt.content.back().type == 1) {
            a = &m.letters[m.FindLetter(pt.content.back())];
        } else if (pt.content.back().type == 2) {
            a = &m.digits[m.FindDigit(pt.content.back())];
        } else if (pt.content.back().type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content.back())];
        }
        GenerateWithGPU(guess, a, pt.max_indices.back());
    }
}

// 异步生成方法
void PriorityQueue::GenerateWithGPUAsync(const string& guess, segment* a, int n, cudaStream_t stream, GPUMemory& gpu_mem) {
    const int GPU_THRESHOLD = 10000;
    if (n <= GPU_THRESHOLD) {
        for (int i = 0; i < n; i++) {
            guesses.emplace_back(guess + a->ordered_values[i]);
        }
        total_guesses += n;
        return;
    }

    // 准备数据
    vector<string>& values = a->ordered_values;
    vector<int> value_lengths(n);
    vector<int> output_offsets(n + 1, 0);
    int total_output_chars = 0;

    for (int i = 0; i < n; i++) {
        value_lengths[i] = values[i].length();
        output_offsets[i + 1] = output_offsets[i] + guess.length() + value_lengths[i];
    }
    total_output_chars = output_offsets[n];

    // 扁平化values
    string flat_values;
    for (int i = 0; i < n; i++) {
        flat_values += values[i];
    }

    // 保存到GPU内存结构
    gpu_mem.n = n;
    gpu_mem.total_output_chars = total_output_chars;
    gpu_mem.output_offsets = output_offsets;

    // 分配设备内存
    cudaMalloc(&gpu_mem.d_guess, guess.size());
    cudaMalloc(&gpu_mem.d_values, flat_values.size());
    cudaMalloc(&gpu_mem.d_value_lengths, n * sizeof(int));
    cudaMalloc(&gpu_mem.d_output_offsets, (n + 1) * sizeof(int));
    cudaMalloc(&gpu_mem.d_output, total_output_chars);
    gpu_mem.h_output = new char[total_output_chars];

    // 异步拷贝数据到设备
    cudaMemcpyAsync(gpu_mem.d_guess, guess.c_str(), guess.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gpu_mem.d_values, flat_values.c_str(), flat_values.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gpu_mem.d_value_lengths, value_lengths.data(), n * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gpu_mem.d_output_offsets, output_offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);

    // 启动核函数
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    generateGuessesKernel<<<gridSize, blockSize, 0, stream>>>(
        gpu_mem.d_guess, guess.length(),
        gpu_mem.d_values, gpu_mem.d_value_lengths, n,
        gpu_mem.d_output, gpu_mem.d_output_offsets
    );

    // 异步拷贝结果回主机
    cudaMemcpyAsync(gpu_mem.h_output, gpu_mem.d_output, total_output_chars, cudaMemcpyDeviceToHost, stream);
}

// 批量生成方法
void PriorityQueue::GenerateBatchWithGPUAsync(const vector<PT>& batch, cudaStream_t stream) {
    // 准备批量数据
    vector<string> prefixes;
    vector<segment*> segments;
    vector<int> counts;

    // 收集每个PT的数据
    for (const auto& pt : batch) {
        // 获取前缀
        string prefix = "";
        if (pt.content.size() > 1) {
            int seg_idx = 0;
            for (int idx : pt.curr_indices) {
                if (pt.content[seg_idx].type == 1) {
                    prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
                } else if (pt.content[seg_idx].type == 2) {
                    prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
                } else if (pt.content[seg_idx].type == 3) {
                    prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
                }
                seg_idx++;
                if (seg_idx == pt.content.size() - 1) break;
            }
        }
        prefixes.push_back(prefix);

        // 获取最后一个segment
        segment* seg_ptr = nullptr;
        if (pt.content.back().type == 1) {
            seg_ptr = &m.letters[m.FindLetter(pt.content.back())];
        } else if (pt.content.back().type == 2) {
            seg_ptr = &m.digits[m.FindDigit(pt.content.back())];
        } else if (pt.content.back().type == 3) {
            seg_ptr = &m.symbols[m.FindSymbol(pt.content.back())];
        }
        segments.push_back(seg_ptr);
        
        // 记录每个PT的猜测数量
        int count = pt.max_indices.back();
        counts.push_back(count);
    }

    // 为每个PT创建GPU内存结构
    for (size_t i = 0; i < batch.size(); i++) {
        GPUMemory gpu_mem;
        GenerateWithGPUAsync(prefixes[i], segments[i], counts[i], stream, gpu_mem);
        gpu_memories.push_back(gpu_mem);
    }
}

// 处理GPU结果
void PriorityQueue::FinalizeGPUResults(cudaStream_t stream) {
    // 等待所有异步操作完成
    cudaStreamSynchronize(stream);
    
    // 处理每个GPU内存结构
    for (auto& gpu_mem : gpu_memories) {
        // 构建结果字符串
        for (int i = 0; i < gpu_mem.n; i++) {
            int start = gpu_mem.output_offsets[i];
            int end = gpu_mem.output_offsets[i+1];
            guesses.emplace_back(gpu_mem.h_output + start, end - start);
        }
        
        // 释放资源
        cudaFree(gpu_mem.d_guess);
        cudaFree(gpu_mem.d_values);
        cudaFree(gpu_mem.d_value_lengths);
        cudaFree(gpu_mem.d_output_offsets);
        cudaFree(gpu_mem.d_output);
        delete[] gpu_mem.h_output;
    }
    gpu_memories.clear();
}

// 修改后的批量处理方法
void PriorityQueue::PopNextBatch(int batch_size) {
    vector<PT> batch;
    
    // 从优先队列中取出batch_size个PT
    for (int i = 0; i < batch_size && !priority.empty(); i++) {
        batch.push_back(priority.front());
        priority.erase(priority.begin());
    }

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // 启动GPU生成当前批次的猜测（异步）
    GenerateBatchWithGPUAsync(batch, stream);

    // 在GPU执行的同时，CPU处理新PT的生成
    vector<PT> new_pts_all;
    for (auto& pt : batch) {
        vector<PT> new_pts = pt.NewPTs();
        new_pts_all.insert(new_pts_all.end(), new_pts.begin(), new_pts.end());
    }
    
    // 插入新PT到优先队列
    for (PT& new_pt : new_pts_all) {
        CalProb(new_pt);
        auto it = priority.begin();
        while (it != priority.end() && it->prob > new_pt.prob) {
            ++it;
        }
        priority.insert(it, new_pt);
    }
    
    // 等待GPU完成并处理结果
    FinalizeGPUResults(stream);
    
    // 清理流
    cudaStreamDestroy(stream);
    
    total_guesses += batch.size();
}
