#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
using namespace std;
using namespace chrono;

// 编译指令如下：
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o test.exe


// 通过这个函数，你可以验证你实现的SIMD哈希函数的正确性
int main()
{
    bit32 state[4];
    MD5Hash("abcd", state);
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << state[i1];
    }
    cout << endl;

    MD5Hash("123", state);
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << state[i1];
    }
    cout << endl;

    MD5Hash("b", state);
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << state[i1];
    }
    cout << endl;

    MD5Hash("@#", state);
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << state[i1];
    }
    cout << endl;

    MD5Hash("a654321", state);
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << state[i1];
    }
    cout << endl;

    MD5Hash("!!23345", state);
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << state[i1];
    }
    cout << endl;

    MD5Hash("b65@#", state);
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << state[i1];
    }
    cout << endl;

    MD5Hash("@#123aa", state);
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << state[i1];
    }
    cout << endl<<endl;

    vector<bit32*> states(4);
    // 为每个 state 分配内存，大小为 4（对应 uint32x4_t 的 4 个 32 位整数）
    for (int i = 0; i < 4; ++i) {
        states[i] = new bit32[8]; // 每个 state 包含 4 个 32 位整数
    }
    vector<string> inputs(8);
    // inputs[0] = "bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva";
    // inputs[1] = "wdhbewbfecbeocberybuvocaehrbyveahocvaehrvbliweufhiaebciebvouiayobevoiaewnvpiaweufboewbqpiconcioenveiruvnqqrevgefvfaaedfwhryyukukiiusrfavrvarwdvdfbyrsjxrshrtsgbwefcwrvkjmgektjbbrnijitnerivbersbpaorjveiruvneiepjarunvpeinveniunvrungprvaregsdghbsgdbhgnhfjssthsegvstrrstsshtraaaaaaaegwwvbtegneshthtntrhyjdyqpkamzwiesjnxeiedhcbyeyrgbiubaibfuaybiuabrhobvabvebvibvoirergmcrefrgxfyiwxypqzoufjrifexmbferxbueryxmbry";
    // inputs[2] = "afaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxafaertcerrecrewxefse";
    // inputs[3] = "gtrymiujuiuiujiutaxexfeferferyhtrrbecrercwdrrfgtrymiujuiuiujiutaxexfeferferyhtrrbecrercwdrrfgtrymiujuiuiujiutaxexfeferferyhtrrbecrercwdrrfgtrymiujuiuiujiutaxexfeferferyhtrrbecrercwdrrfgtrymiujuiuiujiutaxexfeferferyhtrrbecrercwdrrfgtrymiujuiuiujiutaxexfeferferyhtrrbecrercwdrrfgtrymiujuiuiujiutaxexfeferferyhtrrbecrercwdrrfgtrymiujuiuiujiutaxexfeferferyhtrrbecrercwdrrfgtrymiujuiuiujiutaxexfeferferyhtrrbe";

    inputs[0] = "abcd";
    inputs[1] = "123";
    inputs[2] = "b";
    inputs[3] = "@#";
    inputs[4] = "a654321";
    inputs[5] = "!!23345";
    inputs[6] = "b65@#";
    inputs[7] = "@#123aa";


    MD5HashSIMD8(inputs, states);

    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << states[i1][0];
    }
    cout << endl;
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << states[i1][1];
    }
    cout << endl;
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << states[i1][2];
    }
    cout << endl;
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << states[i1][3];
    }
    cout << endl;
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << states[i1][4];
    }
    cout << endl;
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << states[i1][5];
    }
    cout << endl;
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << states[i1][6];
    }
    cout << endl;
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << states[i1][7];
    }
    cout << endl;
    
}
