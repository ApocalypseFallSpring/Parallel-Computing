#include "PCFG.h"
#include <pthread.h>
#include <vector>
#include <string>
#include <iostream>
#include <mutex>
#include <mpi.h>
#include <cstring>
using namespace std;

// // ==================== 单segment场景 ====================
// struct SingleThreadArgs {
//     segment* seg_ptr;    // 指向目标segment
//     int start_idx;       // 起始处理索引
//     int end_idx;         // 结束处理索引（不包含）
//     std::string* output; // 直接写入的目标内存区域
// };

// void* single_thread_func(void* arg) {
//     SingleThreadArgs* args = static_cast<SingleThreadArgs*>(arg);
//     for (int i = args->start_idx; i < args->end_idx; ++i) {
//         // 直接拷贝无需拼接
//         args->output[i - args->start_idx] = args->seg_ptr->ordered_values[i];
//     }
//     return nullptr;
// }

// // ==================== 多segment场景 ====================
// struct MultiThreadArgs {
//     segment* seg_ptr;    // 指向最后一个segment
//     std::string prefix;  // 已生成的前缀字符串
//     int start_idx;
//     int end_idx;
//     std::string* output;
// };

// void* multi_thread_func(void* arg) {
//     MultiThreadArgs* args = static_cast<MultiThreadArgs*>(arg);
//     const std::string& prefix = args->prefix;
//     for (int i = args->start_idx; i < args->end_idx; ++i) {
//         // 执行前缀拼接
//         args->output[i - args->start_idx] = prefix + args->seg_ptr->ordered_values[i];
//     }
//     return nullptr;
// }

// 线程参数结构体（无需存储本地结果）
struct ThreadArgs {
    segment* a;          // 指向当前segment
    std::string prefix;  // 前缀字符串
    int start;           // 当前线程处理的起始索引
    int end;             // 结束索引（不包含）
    std::string* output; // 直接写入的全局内存位置
};

// 线程处理函数（直接写入目标内存）
void* thread_generate(void* arg) {
    ThreadArgs* args = static_cast<ThreadArgs*>(arg);
    for (int i = args->start; i < args->end; ++i) {
        // 直接写入预先分配的内存位置
        args->output[i - args->start] = args->prefix + args->a->ordered_values[i];
    }
    return nullptr;
}

// // 线程参数结构体
// struct ThreadArgs {
//     int start;  //当前线程处理的起始索引（包含）
//     int end;    //当前线程处理的结束索引（不包含）
//     segment* a;
//     bool is_single_segment; //单个segment还是多个，如果是多个->有前缀
//     std::string prefix; //多段的前缀

//     //以下指向主猜测结果
//     std::vector<std::string>* guesses;
//     int* total_guesses;

//     pthread_mutex_t* mutex;
// };

// // 线程函数
// void* generate_guesses_thread(void* arg) {
//     ThreadArgs* args = (ThreadArgs*)arg;

//     //临时存放当前线程生成的猜测，避免频繁加锁
//     std::vector<std::string> local_guesses;
//     int local_count = 0;
    
//     for (int i = args->start; i < args->end; ++i) {
//         std::string guess;
//         if (args->is_single_segment) {
//             guess = args->a->ordered_values[i];
//         } else {
//             guess = args->prefix + args->a->ordered_values[i];
//         }
//         local_guesses.push_back(guess);
//         local_count++;
//     }
    
//     // 合并结果到主猜测结果
//     pthread_mutex_lock(args->mutex);
//     args->guesses->insert(args->guesses->end(), local_guesses.begin(), local_guesses.end());
//     *(args->total_guesses) += local_count;
//     pthread_mutex_unlock(args->mutex);
    
//     return NULL;
// }

// 线程参数结构体
// struct ThreadArgs {
//     int start;
//     int end;
//     segment* a;
//     bool is_single_segment;
//     std::string prefix;
//     std::vector<std::string>* guesses;
//     std::atomic<int>* total_guesses; // 改为原子指针
//     std::mutex* vec_mutex;           // 仅保护vector插入
// };

// // 线程函数
// void* generate_guesses_thread(void* arg) {
//     ThreadArgs* args = static_cast<ThreadArgs*>(arg);
//     vector<string> local_guesses;
//     const int batch_size = 1000; // 根据实际测试调整批量大小
//     int local_count = 0;

//     for (int i = args->start; i < args->end; ++i) {
//         // ...生成guess代码...
//         string guess;
//         if (args->is_single_segment) {
//             guess = args->a->ordered_values[i];
//         } else {
//             guess = args->prefix + args->a->ordered_values[i];
//         }
        
//         local_guesses.emplace_back(guess);
//         local_count++;

//         // 批量提交
//         if (local_guesses.size() >= batch_size) {
//             lock_guard<mutex> lock(*args->vec_mutex);
//             args->guesses->insert(args->guesses->end(), 
//                                  local_guesses.begin(),
//                                  local_guesses.end());
//             local_guesses.clear();
//         }
//     }

//     // 提交剩余数据
//     if (!local_guesses.empty()) {
//         lock_guard<mutex> lock(*args->vec_mutex);
//         args->guesses->insert(args->guesses->end(),
//                             local_guesses.begin(),
//                             local_guesses.end());
//     }

//     // 原子操作无需锁
//     args->total_guesses->fetch_add(local_count);
    
//     return nullptr;
// }

// // 定义线程参数结构体
// struct ThreadArgs {
//     vector<string> *ordered_values; // 指向模型中的值
//     vector<string> *guesses;        // 指向存储猜测的容器
//     int start;                      // 当前线程处理的起始索引
//     int end;                        // 当前线程处理的结束索引
//     mutex *mtx;                     // 用于保护共享资源的互斥锁
//     int *total_guesses;             // 指向总猜测计数器
// };

// // 线程函数
// void *GenerateGuesses(void *args) {
//     ThreadArgs *threadArgs = (ThreadArgs *)args;

//     for (int i = threadArgs->start; i < threadArgs->end; ++i) {
//         string guess = (*threadArgs->ordered_values)[i];

//         // 使用互斥锁保护共享资源
//         threadArgs->mtx->lock();
//         threadArgs->guesses->emplace_back(guess);
//         (*threadArgs->total_guesses)++;
//         threadArgs->mtx->unlock();
//     }

//     return nullptr;
// }

// // 新的线程参数结构体，需包含guess前缀
// struct ThreadArgsWithPrefix {
//     vector<string> *ordered_values;
//     vector<string> *guesses;
//     string prefix;
//     int start;
//     int end;
//     mutex *mtx;
//     int *total_guesses;
// };

// // 新的线程函数
// auto GenerateGuessesWithPrefix = [](void *args) -> void* {
//     ThreadArgsWithPrefix *threadArgs = (ThreadArgsWithPrefix *)args;
//     for (int i = threadArgs->start; i < threadArgs->end; ++i) {
//         string temp = threadArgs->prefix + (*threadArgs->ordered_values)[i];
//         threadArgs->mtx->lock();
//         threadArgs->guesses->emplace_back(temp);
//         (*threadArgs->total_guesses)++;
//         threadArgs->mtx->unlock();
//     }
//     return nullptr;
// };

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
        if (pt.content[index].type == 1)//type==letter
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
        if (pt.content[index].type == 2)//typr==digit
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)//type==symbol
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

// void PriorityQueue::PopNext()
// {

//     // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
//     Generate(priority.front());

//     // 然后需要根据即将出队的PT，生成一系列新的PT
//     vector<PT> new_pts = priority.front().NewPTs();
//     for (PT pt : new_pts)
//     {
//         // 计算概率
//         CalProb(pt);
//         // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
//         for (auto iter = priority.begin(); iter != priority.end(); iter++)
//         {
//             // 对于非队首和队尾的特殊情况
//             if (iter != priority.end() - 1 && iter != priority.begin())
//             {
//                 // 判定概率
//                 if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
//                 {
//                     priority.emplace(iter + 1, pt);
//                     break;
//                 }
//             }
//             if (iter == priority.end() - 1)
//             {
//                 priority.emplace_back(pt);
//                 break;
//             }
//             if (iter == priority.begin() && iter->prob < pt.prob)
//             {
//                 priority.emplace(iter, pt);
//                 break;
//             }
//         }
//     }

//     // 现在队首的PT善后工作已经结束，将其出队（删除）
//     priority.erase(priority.begin());
// }

void PriorityQueue::PopNext()
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 主进程一次处理多个PT
    if (rank == 0) {
        // 一次性取出多个PT（最多取size-1个）
        vector<PT> popped_pts;
        int pop_count = min(static_cast<int>(priority.size()), size);
        for (int i = 0; i < pop_count; i++) {
            popped_pts.push_back(priority.front());
            priority.erase(priority.begin());
        }

        // 分发PT给各进程
        vector<int> pt_counts(size, 0);
        vector<vector<PT>> new_pts_list(size);
        
        // 使用OpenMP并行处理每个PT
        #pragma omp parallel for
        for (int i = 0; i < pop_count; i++) {
            Generate(popped_pts[i]);
            vector<PT> new_pts = popped_pts[i].NewPTs();
            new_pts_list[i] = new_pts;
            pt_counts[i] = new_pts.size();
        }

        // 收集其他进程的新PT
        for (int r = 1; r < size; r++) {
            int count;
            MPI_Recv(&count, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pt_counts[r] = count;
            
            if (count > 0) {
                new_pts_list[r].resize(count);
                MPI_Recv(new_pts_list[r].data(), count * sizeof(PT), MPI_BYTE, 
                         r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        // 合并所有新PT到优先队列
        #pragma omp parallel for
        for (int r = 0; r < size; r++) {
            for (PT& pt : new_pts_list[r]) {
                CalProb(pt);
                // 插入新PT到优先队列（保持降序）
                auto it = priority.begin();
                while (it != priority.end() && it->prob > pt.prob) {
                    ++it;
                }
                #pragma omp critical
                {
                    priority.insert(it, pt);
                }
            }
        }
    }
    // 从进程处理分配的PT
    else {
        if (!priority.empty()) {
            PT pt = priority.front();
            priority.erase(priority.begin());
            
            Generate(pt);
            vector<PT> new_pts = pt.NewPTs();
            
            // 发送新PT回主进程
            int count = new_pts.size();
            MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            
            if (count > 0) {
                MPI_Send(new_pts.data(), count * sizeof(PT), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            }
        }
    }
}


// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
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


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
// void PriorityQueue::Generate(PT pt)
// {
//     // 计算PT的概率，这里主要是给PT的概率进行初始化
//     CalProb(pt);

//     // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
//     if (pt.content.size() == 1)
//     {
//         // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
//         segment *a;
//         // 在模型中定位到这个segment
//         if (pt.content[0].type == 1)//letter
//         {
//             a = &m.letters[m.FindLetter(pt.content[0])];
//         }
//         if (pt.content[0].type == 2)//digit
//         {
//             a = &m.digits[m.FindDigit(pt.content[0])];
//         }
//         if (pt.content[0].type == 3)//symbol
//         {
//             a = &m.symbols[m.FindSymbol(pt.content[0])];
//         }
//         // Multi-thread TODO：
//         // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
//         // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
//         // 这个过程是可以高度并行化的
//         // for (int i = 0; i < pt.max_indices[0]; i += 1)
//         // {
//         //     string guess = a->ordered_values[i];
//         //     // cout << guess << endl;
//         //     guesses.emplace_back(guess);
//         //     total_guesses += 1;
//         // }

//         if(a->ordered_values.size() < 10000)
//         {
//             // 如果这个segment的value数量小于1000，直接串行处理
//             for (int i = 0; i < pt.max_indices[0]; i += 1)
//             {
//                 string guess = a->ordered_values[i];
//                 guesses.emplace_back(guess);
//                 total_guesses += 1;
//             }
//             return;
//         }

//         // 预计算总任务量
//         const int n = pt.max_indices[0];
//         const size_t original_size = guesses.size();
        
//         // 预分配全局内存
//         guesses.resize(original_size + n);
        
//         // 计算每个线程的任务块
//         const int thread_num = 6;
//         pthread_t threads[thread_num];
//         ThreadArgs args[thread_num];
//         const int chunk_size = (n + thread_num - 1) / thread_num;

//         // 创建线程（直接写入guesses的预留空间）
//         for (int t = 0; t < thread_num; ++t) {
//             const int start = t * chunk_size;
//             const int end = std::min(start + chunk_size, n);
            
//             args[t].a = a;
//             args[t].prefix = "";
//             args[t].start = start;
//             args[t].end = end;
//             args[t].output = &guesses[original_size + start]; // 指向写入起始位置
            
//             pthread_create(&threads[t], nullptr, thread_generate, &args[t]);
//         }

//         // 等待所有线程完成
//         for (int t = 0; t < thread_num; ++t) {
//             pthread_join(threads[t], nullptr);
//         }

//         total_guesses += n;
  
//         // #pragma omp parallel for
//         // for (int i = 0; i < pt.max_indices[0]; i++)
//         // {
//         //     string guess = a->ordered_values[i];
//         //     #pragma omp critical
//         //     {
//         //         guesses.emplace_back(guess);
//         //         total_guesses += 1;
//         //     }
//         // }

//         // #pragma omp parallel
//         // {
//         //     vector<string> local_guesses;  // 线程本地容器
//         //     int local_count = 0;           // 线程本地计数器

//         //     #pragma omp for nowait         // nowait消除隐式同步
//         //     for (int i = 0; i < pt.max_indices[0]; i++) {
//         //         local_guesses.emplace_back(a->ordered_values[i]);
//         //         local_count++;
//         //     }

//         //     #pragma omp critical          // 合并结果时仅同步一次
//         //     {
//         //         guesses.insert(guesses.end(), local_guesses.begin(), local_guesses.end());
//         //         total_guesses += local_count;
//         //     }
//         // }


//         // // 并行生成猜测
//         // const size_t n = pt.max_indices[0];
//         // const size_t current_size = guesses.size();
//         // guesses.resize(current_size + n);

//         // #pragma omp parallel for
//         // for (int i = 0; i < n; ++i) {
//         //     guesses[current_size + i] = a->ordered_values[i];
//         // }

//         // #pragma omp atomic
//         // total_guesses += n;

//     }
//     else
//     {
//         string guess;
//         int seg_idx = 0;
//         // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
//         // segment值根据curr_indices中对应的值加以确定
//         // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
//         for (int idx : pt.curr_indices)
//         {
//             if (pt.content[seg_idx].type == 1)
//             {
//                 guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
//             }
//             if (pt.content[seg_idx].type == 2)
//             {
//                 guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
//             }
//             if (pt.content[seg_idx].type == 3)
//             {
//                 guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
//             }
//             seg_idx += 1;
//             if (seg_idx == pt.content.size() - 1)
//             {
//                 break;
//             }
//         }

//         // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
//         segment *a;
//         if (pt.content[pt.content.size() - 1].type == 1)
//         {
//             a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
//         }
//         if (pt.content[pt.content.size() - 1].type == 2)
//         {
//             a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
//         }
//         if (pt.content[pt.content.size() - 1].type == 3)
//         {
//             a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
//         }

//         //std::cout<<a->ordered_values.size()<<std::endl;
        
//         // Multi-thread TODO：
//         // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
//         // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
//         // 这个过程是可以高度并行化的
//         // for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
//         // {
//         //     string temp = guess + a->ordered_values[i];
//         //     // cout << temp << endl;
//         //     guesses.emplace_back(temp);
//         //     total_guesses += 1;
//         // }

//         if (a->ordered_values.size() < 10000)
//         {
//             // 如果当前segment的value数量小于5000，直接使用单线程生成
//             for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
//             {
//                 string temp = guess + a->ordered_values[i];
//                 guesses.emplace_back(temp);
//                 total_guesses += 1;
//             }
//             return;
//         }

//         const int n = pt.max_indices[pt.content.size()-1];
//         const size_t original_size = guesses.size();
        
//         guesses.resize(original_size + n);

//         const int thread_num = 6;
//         pthread_t threads[thread_num];
//         ThreadArgs args[thread_num];
//         const int chunk_size = (n + thread_num - 1) / thread_num;

//         for (int t = 0; t < thread_num; ++t) {
//             const int start = t * chunk_size;
//             const int end = std::min(start + chunk_size, n);
            
//             args[t].a = a;
//             args[t].prefix = guess;
//             args[t].start = start;
//             args[t].end = end;
//             args[t].output = &guesses[original_size + start];
            
//             pthread_create(&threads[t], nullptr, thread_generate, &args[t]);
//         }

//         for (int t = 0; t < thread_num; ++t) {
//             pthread_join(threads[t], nullptr);
//         }

//         total_guesses += n;

//         // #pragma omp parallel for
//         // for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i++)
//         // {
//         //     string temp = guess + a->ordered_values[i];
//         //     #pragma omp critical
//         //     {
//         //         guesses.emplace_back(temp);
//         //         total_guesses += 1;
//         //     }
//         // }

//         // #pragma omp parallel  // 创建并行区域
//         // {
//         //     // 每个线程创建本地缓冲容器和计数器
//         //     std::vector<std::string> local_guesses;
//         //     int local_count = 0;

//         //     // 获取固定前缀的副本（避免多线程下多次访问共享变量）
//         //     const std::string local_guess = guess;

//         //     // 分割循环迭代到不同线程，nowait取消隐式同步
//         //     #pragma omp for nowait
//         //     for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i++) {
//         //         // 本地操作无需同步
//         //         local_guesses.emplace_back(local_guess + a->ordered_values[i]);
//         //         local_count++;
//         //     }

//         //     // 合并本地结果到全局容器（仅需一次同步）
//         //     #pragma omp critical
//         //     {
//         //         // 预分配空间减少内存重分配
//         //         guesses.reserve(guesses.size() + local_guesses.size());
//         //         // 使用移动语义避免字符串拷贝
//         //         guesses.insert(guesses.end(), 
//         //                     std::make_move_iterator(local_guesses.begin()),
//         //                     std::make_move_iterator(local_guesses.end()));
//         //         // 原子操作更新计数器
//         //         #pragma omp atomic
//         //         total_guesses += local_count;
//         //     }
//         // }

//         // #pragma omp parallel
//         // {
//         //     vector<string> local_guesses;  // 线程本地容器
//         //     int local_count = 0;           // 线程本地计数器

//         //     const std::string local_guess = guess;

//         //     #pragma omp for nowait
//         //     for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i++) {
//         //         local_guesses.emplace_back(local_guess + a->ordered_values[i]);
//         //         local_count++;
//         //     }

//         //     #pragma omp critical
//         //     {
//         //         // 预分配空间减少内存重分配
//         //         guesses.reserve(guesses.size() + local_guesses.size());
//         //         // 使用移动语义避免字符串拷贝
//         //         guesses.insert(guesses.end(), 
//         //                     std::make_move_iterator(local_guesses.begin()),
//         //                     std::make_move_iterator(local_guesses.end()));

//         //         #pragma omp atomic
//         //         total_guesses += local_count;
//         //     }
//         // }

//         // // 并行生成猜测
//         // const size_t n = pt.max_indices[pt.content.size() - 1];
//         // const size_t current_size = guesses.size();
//         // guesses.resize(current_size + n);
//         // const string base_guess = guess; // 线程安全的副本

//         // #pragma omp parallel for
//         // for (int i = 0; i < n; ++i) {
//         //     guesses[current_size + i] = base_guess + a->ordered_values[i];
//         // }

//         // #pragma omp atomic
//         // total_guesses += n;

//     }
// }

void PriorityQueue::Generate(PT pt)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 主进程计算概率
    if (rank == 0) {
        CalProb(pt);
    }

    segment* a = nullptr;
    string guess_prefix = "";
    int scene = (pt.content.size() == 1) ? 0 : 1;
    int n = 0;
    int seg_type = 0;
    int model_index = -1;

    // 主进程准备数据
    if (rank == 0) {
        if (scene == 0) {
            seg_type = pt.content[0].type;
            if (seg_type == 1) {
                model_index = m.FindLetter(pt.content[0]);
                if (model_index != -1) a = &m.letters[model_index];
            } else if (seg_type == 2) {
                model_index = m.FindDigit(pt.content[0]);
                if (model_index != -1) a = &m.digits[model_index];
            } else if (seg_type == 3) {
                model_index = m.FindSymbol(pt.content[0]);
                if (model_index != -1) a = &m.symbols[model_index];
            }
            n = pt.max_indices[0];
        } else {
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
            int last_idx = pt.content.size() - 1;
            seg_type = pt.content[last_idx].type;
            if (seg_type == 1) {
                model_index = m.FindLetter(pt.content[last_idx]);
                if (model_index != -1) a = &m.letters[model_index];
            } else if (seg_type == 2) {
                model_index = m.FindDigit(pt.content[last_idx]);
                if (model_index != -1) a = &m.digits[model_index];
            } else if (seg_type == 3) {
                model_index = m.FindSymbol(pt.content[last_idx]);
                if (model_index != -1) a = &m.symbols[model_index];
            }
            n = pt.max_indices[last_idx];
        }
    }

    // 广播基础信息
    MPI_Bcast(&scene, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&seg_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&model_index, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 检查segment是否有效
    if (model_index == -1 || n == 0) {
        if (rank == 0) {
            cerr << "Error: Segment not found in model or zero tasks" << endl;
        }
        return;
    }

    // 所有进程获取segment引用
    segment* local_a = nullptr;
    if (seg_type == 1 && model_index < m.letters.size()) {
        local_a = &m.letters[model_index];
    } else if (seg_type == 2 && model_index < m.digits.size()) {
        local_a = &m.digits[model_index];
    } else if (seg_type == 3 && model_index < m.symbols.size()) {
        local_a = &m.symbols[model_index];
    }

    if (!local_a) {
        if (rank == 0) {
            cerr << "Error: Invalid segment reference" << endl;
        }
        return;
    }

    // 广播前缀（多segment场景）
    if (scene == 1) {
        int prefix_len = 0;
        if (rank == 0) {
            prefix_len = guess_prefix.size();
        }
        MPI_Bcast(&prefix_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (prefix_len > 0) {
            char* prefix_buf = new char[prefix_len + 1];
            if (rank == 0) {
                strcpy(prefix_buf, guess_prefix.c_str());
            }
            MPI_Bcast(prefix_buf, prefix_len, MPI_CHAR, 0, MPI_COMM_WORLD);
            prefix_buf[prefix_len] = '\0';
            guess_prefix = string(prefix_buf);
            delete[] prefix_buf;
        }
    }

    // 任务划分
    int chunk_size = (n + size - 1) / size;
    int start = rank * chunk_size;
    int end = min(start + chunk_size, n);
    int local_count = max(0, end - start);

    // 本地生成猜测 - 使用OpenMP并行化
    vector<string> local_guesses;
    local_guesses.resize(local_count);
    
    #pragma omp parallel for
    for (int i = 0; i < local_count; i++) {
        int global_idx = start + i;
        if (scene == 0) {
            local_guesses[i] = local_a->ordered_values[global_idx];
        } else {
            local_guesses[i] = guess_prefix + local_a->ordered_values[global_idx];
        }
    }

    // 收集所有进程的猜测数量
    vector<int> all_counts(size, 0);
    MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 收集字符串长度信息
    vector<int> local_lengths(local_count);
    #pragma omp parallel for
    for (int i = 0; i < local_count; i++) {
        local_lengths[i] = local_guesses[i].size();
    }

    vector<int> all_lengths;
    vector<int> displs(size, 0), rcounts(size, 0);
    
    if (rank == 0) {
        int total_lengths = 0;
        for (int i = 0; i < size; i++) {
            rcounts[i] = all_counts[i];
            displs[i] = total_lengths;
            total_lengths += all_counts[i];
        }
        all_lengths.resize(total_lengths);
    }

    MPI_Gatherv(local_lengths.data(), local_count, MPI_INT,
               all_lengths.data(), rcounts.data(), displs.data(), MPI_INT,
               0, MPI_COMM_WORLD);

    // 收集字符串数据
    int total_chars = 0;
    vector<int> char_offsets(local_count + 1, 0);
    for (int i = 0; i < local_count; i++) {
        char_offsets[i+1] = char_offsets[i] + local_lengths[i];
    }
    int local_chars = char_offsets[local_count];
    
    vector<char> local_data(local_chars);
    #pragma omp parallel for
    for (int i = 0; i < local_count; i++) {
        memcpy(&local_data[char_offsets[i]], 
               local_guesses[i].c_str(), 
               local_lengths[i]);
    }

    vector<int> all_chars(size, 0);
    MPI_Gather(&local_chars, 1, MPI_INT, all_chars.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<char> global_data;
    vector<int> char_displs(size, 0), char_rcounts(size, 0);
    
    if (rank == 0) {
        int total_global_chars = 0;
        for (int i = 0; i < size; i++) {
            char_rcounts[i] = all_chars[i];
            char_displs[i] = total_global_chars;
            total_global_chars += all_chars[i];
        }
        global_data.resize(total_global_chars);
    }

    MPI_Gatherv(local_data.data(), local_chars, MPI_CHAR,
               global_data.data(), char_rcounts.data(), char_displs.data(), MPI_CHAR,
               0, MPI_COMM_WORLD);

    // 主进程处理最终结果
    if (rank == 0) {
        int pos = 0;
        for (int i = 0; i < all_lengths.size(); i++) {
            guesses.emplace_back(global_data.data() + pos, all_lengths[i]);
            pos += all_lengths[i];
        }
        total_guesses += all_lengths.size();
    }
}
