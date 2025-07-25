#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <mpi.h>
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv); // 初始化MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "here" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;
    // std::ofstream a("./files/results.txt");
    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();
        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " <<history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            // 在此处更改实验生成的猜测上限
            int generate_n=10000000;
            if (history + q.total_guesses > generate_n)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds"<< endl;
                cout << "Hash time:" << time_hash << "seconds"<<endl;
                cout << "Train time:" << time_train <<"seconds"<<endl;
                break;
            }
        }
        // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
        // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
        if (curr_num > 1000000)
        {
            bit32 state[4];
            auto start_hash = system_clock::now();
            for (string pw : q.guesses)
            {
                // TODO：对于SIMD实验，将这里替换成你的SIMD MD5函数
                MD5Hash(pw, state);

                // 以下注释部分用于输出猜测和哈希，但是由于自动测试系统不太能写文件，所以这里你可以改成cout
                // a<<pw<<"\t";
                // for (int i1 = 0; i1 < 4; i1 += 1)
                // {
                //     a << std::setw(8) << std::setfill('0') << hex << state[i1];
                // }
                // a << endl;
            }  

            // int max_len = 0;   
            // for (string pw : q.guesses)
            // {
            //     max_len = max(max_len, (int)pw.length());
            // }
            // cout << "max_len:" << max_len*8 << endl;


            // vector<bit32*> states(4);
            // // 为每个 state 分配内存，大小为 4（对应 uint32x4_t 的 4 个 32 位整数）
            // for (int i = 0; i < 4; ++i) {
            //     states[i] = new bit32[4]; // 每个 state 包含 4 个 32 位整数
            // }
            // vector<string> inputs(4);
            // auto start_hash = system_clock::now();
            // for  (int i = 0; i < q.guesses.size(); i += 4)
            // {
            //     inputs[0] = q.guesses[i];
            //     inputs[1] = q.guesses[i + 1];
            //     inputs[2] = q.guesses[i + 2];
            //     inputs[3] = q.guesses[i + 3];
            //     MD5HashSIMD(inputs, states);
            // }
            // if (q.guesses.size() % 4 != 0)
            // {
            //     bit32 state[4];
            //     for (int i = q.guesses.size() - q.guesses.size() % 4; i < q.guesses.size(); i++)
            //     {
            //         MD5Hash(q.guesses[i], state);
            //     }
            // }

            // vector<bit32*> states(4);
            // for (int i = 0; i < 4; ++i) {
            //     states[i] = new bit32[2];
            // }
            // vector<string> inputs(2);
            // auto start_hash = system_clock::now();
            // for  (int i = 0; i < q.guesses.size(); i += 2)
            // {
            //     inputs[0] = q.guesses[i];
            //     inputs[1] = q.guesses[i + 1];
            //     MD5HashSIMD2(inputs, states);
            // }
            // if (q.guesses.size() % 2 != 0)
            // {
            //     bit32 state[4];
            //     for (int i = q.guesses.size() - q.guesses.size() % 2; i < q.guesses.size(); i++)
            //     {
            //         MD5Hash(q.guesses[i], state);
            //     }
            // }

            // vector<bit32*> states(4);
            // for (int i = 0; i < 4; ++i) {
            //     states[i] = new bit32[8];
            // }
            // vector<string> inputs(8);
            // auto start_hash = system_clock::now();
            // for  (int i = 0; i < q.guesses.size(); i += 8)
            // {
            //     inputs[0] = q.guesses[i];
            //     inputs[1] = q.guesses[i + 1];
            //     inputs[2] = q.guesses[i + 2];
            //     inputs[3] = q.guesses[i + 3];
            //     inputs[4] = q.guesses[i + 4];
            //     inputs[5] = q.guesses[i + 5];
            //     inputs[6] = q.guesses[i + 6];
            //     inputs[7] = q.guesses[i + 7];
            //     MD5HashSIMD8(inputs, states);
            // }
            // if (q.guesses.size() % 8 != 0)
            // {
            //     bit32 state[4];
            //     for (int i = q.guesses.size() - q.guesses.size() % 8; i < q.guesses.size(); i++)
            //     {
            //         MD5Hash(q.guesses[i], state);
            //     }
            // }
            
            // 在这里对哈希所需的总时长进行计算
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
            

            // 记录已经生成的口令总数
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }

    MPI_Finalize();

    return 0;
}
