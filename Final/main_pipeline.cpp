#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <thread>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <unordered_set>
#include <algorithm>
using namespace std;
using namespace chrono;

// 线程安全的队列，用于存储待哈希的口令批次
class ThreadSafeQueue {
private:
    queue<vector<string>> batches;
    mutex mtx;
    condition_variable cv;
    bool generation_done = false;
    atomic<int> batches_processed{0};
    
public:
    void push(vector<string> batch) {
        lock_guard<mutex> lock(mtx);
        batches.push(move(batch));
        cv.notify_one();
    }

    bool pop(vector<string>& batch) {
        unique_lock<mutex> lock(mtx);
        cv.wait(lock, [this] { return !batches.empty() || generation_done; });
        
        if (batches.empty() && generation_done) 
            return false;
        
        batch = move(batches.front());
        batches.pop();
        return true;
    }

    void set_done() {
        lock_guard<mutex> lock(mtx);
        generation_done = true;
        cv.notify_all();
    }
    
    void increment_processed() {
        batches_processed++;
    }
    
    int get_processed() {
        return batches_processed.load();
    }
};

// 哈希工作线程函数
void hash_worker(ThreadSafeQueue& queue, atomic<int>& cracked_count, 
                 const unordered_set<string>& test_set, double& hash_time) {
    auto start = system_clock::now();
    
    while (true) {
        vector<string> batch;
        if (!queue.pop(batch)) break;
        
        
        vector<bit32*> states(4);
        for (int i = 0; i < 4; ++i) {
            states[i] = new bit32[8];
        }
        vector<string> inputs(8);
        auto start_hash = system_clock::now();
        for  (int i = 0; i < batch.size(); i += 8)
        {
            inputs[0] = batch[i];
            inputs[1] = batch[i + 1];
            inputs[2] = batch[i + 2];
            inputs[3] = batch[i + 3];
            inputs[4] = batch[i + 4];
            inputs[5] = batch[i + 5];
            inputs[6] = batch[i + 6];
            inputs[7] = batch[i + 7];
            if (test_set.find(inputs[0]) != test_set.end()) {
                cracked_count++;
            }
            if (test_set.find(inputs[1]) != test_set.end()) {
                cracked_count++;
            }
            if (test_set.find(inputs[2]) != test_set.end()) {
                cracked_count++;
            }
            if (test_set.find(inputs[3]) != test_set.end()) {
                cracked_count++;
            }
            if (test_set.find(inputs[4]) != test_set.end()) {
                cracked_count++;
            }
            if (test_set.find(inputs[5]) != test_set.end()) {
                cracked_count++;
            }
            if (test_set.find(inputs[6]) != test_set.end()) {
                cracked_count++;
            }
            if (test_set.find(inputs[7]) != test_set.end()) {
                cracked_count++;
            }
            MD5HashSIMD8(inputs, states);
        }
        if (batch.size() % 8 != 0)
        {
            bit32 state[4];
            for (int i = batch.size() - batch.size() % 8; i < batch.size(); i++)
            {
                MD5Hash(batch[i], state);
            }
        }
        
        
        // 对整批口令进行哈希和破解检查
        // for (const string& pw : batch) {
        //     // 检查是否在测试集中
        //     if (test_set.find(pw) != test_set.end()) {
        //         cracked_count++;
        //     }
            
        //     // 计算哈希值
        //     bit32 state[4];
        //     MD5Hash(pw, state);
        // }
        
        queue.increment_processed();
    }
    
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    hash_time += double(duration.count()) * microseconds::period::num / microseconds::period::den;
}

int main()
{
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    atomic<int> cracked_count{0}; // 破解的口令数量
    
    // 加载测试数据集 - 确保路径正确
    unordered_set<string> test_set;
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    if (!test_data) {
        cerr << "Error: Failed to open test data file" << endl;
        return 1;
    }
    
    string pw;
    int test_count = 0;
    while (getline(test_data, pw)) {
        // 移除换行符
        if (!pw.empty() && pw[pw.size()-1] == '\n') {
            pw.erase(pw.size()-1);
        }
        if (!pw.empty() && pw[pw.size()-1] == '\r') {
            pw.erase(pw.size()-1);
        }
        
        test_set.insert(pw);
        test_count++;
        if (test_count >= 1000000) {
            break;
        }
    }
    test_data.close();
    
    cout << "Loaded " << test_set.size() << " test passwords" << endl;
    
    // 验证测试集加载是否正确
    if (test_set.empty()) {
        cerr << "Error: Test set is empty" << endl;
        return 1;
    }
    
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
    cout << "Model training completed in " << time_train << " seconds" << endl;

    q.init();
    cout << "Initialization complete" << endl;
    
    const size_t BATCH_SIZE = 1000000; // 每批口令数量
    ThreadSafeQueue hash_queue;
    
    // 启动哈希工作线程
    double worker_hash_time = 0;
    thread hash_thread(hash_worker, ref(hash_queue), ref(cracked_count), 
                     ref(test_set), ref(worker_hash_time));
    
    int total_generated = 0; // 当前批次生成的口令数
    int all_generated = 0;   // 总共生成的口令数
    auto start_guess = system_clock::now();
    
    const int GENERATE_LIMIT = 10000000;
    
    while (!q.priority.empty())
    {
        // 检查是否达到生成上限
        if (all_generated >= GENERATE_LIMIT) {
            cout << "Reached generation limit: " << GENERATE_LIMIT << endl;
            break;
        }
        
        q.PopNext();
        total_generated = q.guesses.size();
        all_generated = total_generated + (hash_queue.get_processed() * BATCH_SIZE);
        
        // 定期报告进度
        if (all_generated % 100000 == 0) {
            cout << "Guesses generated: " << all_generated 
                 << ", Cracked: " << cracked_count.load() << endl;
        }
        
        // 当积累足够的口令时，提交给哈希线程
        if (q.guesses.size() >= BATCH_SIZE) {
            // 移动当前批次到哈希队列
            vector<string> batch;
            batch.swap(q.guesses);
            hash_queue.push(move(batch));
            cout << "Sent batch to hash worker. Current cracked: " << cracked_count.load() << endl;
            
            // 更新总生成数
            all_generated += batch.size();
        }
    }
    
    // 处理剩余的口令
    if (!q.guesses.empty()) {
        cout << "Sending final batch with " << q.guesses.size() << " passwords" << endl;
        hash_queue.push(move(q.guesses));
        all_generated += q.guesses.size();
    }
    
    cout << "All passwords generated. Total: " << all_generated << endl;
    
    // 通知哈希线程工作完成
    hash_queue.set_done();
    cout << "Set done flag for hash worker" << endl;
    
    auto end_guess = system_clock::now();
    auto duration_guess = duration_cast<microseconds>(end_guess - start_guess);
    time_guess = double(duration_guess.count()) * microseconds::period::num / microseconds::period::den;
    
    // 等待哈希线程完成
    cout << "Waiting for hash worker to finish..." << endl;
    hash_thread.join();
    cout << "Hash worker finished" << endl;
    
    time_hash = worker_hash_time;
    
    // 最终统计报告
    cout << "\n========== Final Statistics ==========" << endl;
    cout << "Guess time: " << time_guess << " seconds" << endl;
    cout << "Hash time: " << time_hash << " seconds" << endl;
    cout << "Train time: " << time_train << " seconds" << endl;
    cout << "Total guesses generated: " << all_generated << endl;
    cout << "Passwords cracked: " << cracked_count.load() << endl;
    
    return 0;
}
