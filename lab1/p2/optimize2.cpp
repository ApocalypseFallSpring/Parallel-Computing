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

void optimize2(int n)
{
    if (n == 1)
        return;
    else{
        for(int i = 0; i < n/2; i++)
            a[i] += a[n-i-1];
        n = n/2;
        optimize2(n);
    }
}

int main()
{
    long long head, tail, freq;
    init(N);
    //similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);

    optimize2(a);

    //end time
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "Col_normal:" << (tail-head) * 1.0 / freq << "ms" << endl;

}
