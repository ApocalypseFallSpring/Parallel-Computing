#include <iostream>
#include <windows.h>
#include <stdlib.h>
using namespace std;

//计算给定n*n矩阵的每一列与给定向量的内积
//2.cache优化

const int N=20000; //matrix size
int* a;
int** b;

void init(int n)
{
    a = new int[N];
    b = new int*[N];
    for(int i = 0; i < N; i++)
    {
        a[i] = i + 1;
        b[i] = new int[N];
        for(int j = 0; j < N; j++)
            b[i][j] = i+j;
    }
}


int* cache(int* a, int** b)
{
    int* result = new int[N];
    for (int i = 0; i < N; i++)
    {
        result[i] = 0;
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            result[j] += a[i] * b[i][j];
        }
    }
    return result;
}

int main()
{
    long long head, tail, freq;
    init(N);
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);

    cache(a,b);

    //end time
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "time:" << (tail-head) * 1.0 / freq << "ms" << endl;

}
