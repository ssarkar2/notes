#include <iostream>
#include <ratio>
#include <cassert>

// The cannonical quantization specification
template <typename Scale, int Zero>
struct QuantSpec {
    using scale = Scale;
    static constexpr int zero = Zero;
};


// Note min/max are kept as integers, but they really are floats in practice
// a different way to specify quantization: via min/max
// We compute scale/zero from min/max and compile time
template <
    int Min, int Max,
    int QMin = -128,
    int QMax = 127
>
struct MinMaxSpec {
    static_assert(Min < Max);

    using scale = std::ratio<Max - Min, QMax - QMin>;
    static constexpr int zero =
        QMin - Min * (QMax - QMin) / (Max - Min);
};

// We expect the template type MM to have scale and zero fields
template <typename MM>
using Canonicalized =
    QuantSpec<typename MM::scale, MM::zero>;

template <typename Quant>
struct Quantized {
    using quant = Quant;
    int value;
};

// construction helper
// construct from scale zero
template <typename Scale, int Zero>
using QuantFromScaleZero = Quantized<QuantSpec<Scale, Zero>>;

// construct from minmax
template <int Min, int Max>
using QuantFromMinMax =
    Quantized<Canonicalized<MinMaxSpec<Min, Max>>>;

// compile time check for quant domain
template <typename Q1, typename Q2>
concept SameQuant =
    std::ratio_equal_v<typename Q1::scale, typename Q2::scale> &&
    Q1::zero == Q2::zero;

// addn
template <typename Q1, typename Q2>
requires SameQuant<Q1, Q2>
auto operator+(const Quantized<Q1>& a,
               const Quantized<Q2>& b)
{
    return Quantized<Q1>{a.value + b.value};
}


// sample:
using S0Z0_SZ = QuantFromScaleZero<std::ratio<1, 255>, 127>;
using S0Z0_MM = QuantFromMinMax<-1, 0>;

using S1Z1_MM = QuantFromMinMax<-2, 2>;


int main() {
    S0Z0_SZ a{10};
    S0Z0_MM b{20};
    // These 2 types are essentially the same (though one is defined using s/z and the other using min/max)
    auto c = a + b;

    S0Z0_SZ a2{20};
    auto c2 = a + a2;
    std::cout << "a.value = " << a.value << ", a2.value = " << a2.value << ", c.value = " << c.value << std::endl;
    std::cout << "a.value = " << a.value << ", a2.value = " << a2.value << ", c2.value = " << c2.value << std::endl;

    assert(c.value == c2.value);

    S1Z1_MM a3{15}; // a different range
    //auto x = a + a3; // should not compile
    return 0;
}