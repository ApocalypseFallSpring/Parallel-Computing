#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2

int main()
{
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


    
    // 加载一些测试数据
    unordered_set<std::string> test_set;
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    int test_count=0;
    string pw;
    while(test_data>>pw)
    {   
        test_count+=1;
        test_set.insert(pw);
        if (test_count>=1000000)
        {
            break;
        }
    }
    int cracked=0;

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
            int generate_n = 10000000;
            if (history + q.total_guesses > generate_n)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds"<< endl;
                cout << "Hash time:" << time_hash << "seconds"<<endl;
                cout << "Train time:" << time_train <<"seconds"<<endl;
                cout<<"Cracked:"<< cracked<<endl;
                break;
            }
        }
        // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
        // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
        if (curr_num > 1000000)
        {
            vector<bit32*> states(4);
            for (int i = 0; i < 4; ++i) {
                states[i] = new bit32[8];
            }
            vector<string> inputs(8);
            auto start_hash = system_clock::now();
            for  (int i = 0; i < q.guesses.size(); i += 8)
            {
                inputs[0] = q.guesses[i];
                inputs[1] = q.guesses[i + 1];
                inputs[2] = q.guesses[i + 2];
                inputs[3] = q.guesses[i + 3];
                inputs[4] = q.guesses[i + 4];
                inputs[5] = q.guesses[i + 5];
                inputs[6] = q.guesses[i + 6];
                inputs[7] = q.guesses[i + 7];
                if (test_set.find(inputs[0]) != test_set.end()) {
                    cracked+=1;
                }
                if (test_set.find(inputs[1]) != test_set.end()) {
                    cracked+=1;
                }
                if (test_set.find(inputs[2]) != test_set.end()) {
                    cracked+=1;
                }
                if (test_set.find(inputs[3]) != test_set.end()) {
                    cracked+=1;
                }
                if (test_set.find(inputs[4]) != test_set.end()) {
                    cracked+=1;
                }
                if (test_set.find(inputs[5]) != test_set.end()) {
                    cracked+=1;
                }
                if (test_set.find(inputs[6]) != test_set.end()) {
                    cracked+=1;
                }
                if (test_set.find(inputs[7]) != test_set.end()) {
                    cracked+=1;
                }
                MD5HashSIMD8(inputs, states);
            }
            if (q.guesses.size() % 8 != 0)
            {
                bit32 state[4];
                for (int i = q.guesses.size() - q.guesses.size() % 8; i < q.guesses.size(); i++)
                {
                    MD5Hash(q.guesses[i], state);
                }
            }
            
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
}
