#include <iostream>
#include <windows.h>
#include <omp.h>
#include <thread>
using namespace std;

const int N = 1024*1024*1024;
int* a;

void init(int n)
{
    a = new int[N];
    for(int i = 0; i < N; i++)
    {
        a[i] = 1;
    }
}

long long normal(int* a)
{
    long long result = 0;
    for(int i = 0; i < N; i++)
    {
        result += a[i];
    }
    return result;
}


int main()
{
    long long head, tail, freq;
    init(N);
    //similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);

    normal(a);

    //end time
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "time:" << (tail-head) * 1.0 / freq << "ms" << endl;

}
