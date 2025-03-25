#include <iostream>
#include <windows.h>
#include <stdlib.h>
using namespace std;

//�������n*n�����ÿһ��������������ڻ�
//1.���з���

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

int* normal(int* a, int** b)
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
            result[i] += a[j] * b[j][i];
        }
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

    normal(a,b);

    //end time
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "time:" << (tail-head) * 1.0 / freq << "ms" << endl;
}
