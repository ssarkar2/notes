#include <iostream>
#include <ratio>
#include <cassert>

template <int NBits = 8, bool Symmetric = true>
struct QuantSpec {
    static constexpr int qmin = Symmetric ? (-(1 << (NBits - 1)) + 1) : 0;
    static constexpr int qmax = Symmetric ? ((1 << (NBits - 1)) - 1) : ((1 << NBits) - 1);
    static_assert(qmax > qmin, "QMax must be greater than QMin");
};

template <float Scale, int Zero, int NBits = 8, bool Symmetric = true>
struct QuantSpecScaleZero : public QuantSpec<NBits, Symmetric> {
    static constexpr float scale = Scale;
    static constexpr int zero = Zero;
    static_assert(zero >= QuantSpec<NBits, Symmetric>::qmin && zero <= QuantSpec<NBits, Symmetric>::qmax, "Zero must be between QMin and QMax");
};

template <float Min, float Max, int NBits = 8, bool Symmetric = true>
struct QuantSpecMinMax : public QuantSpec<NBits, Symmetric> {
    static_assert(Max > Min, "Max must be greater than Min");
    static constexpr float scale = (Max - Min) / (QuantSpec<NBits, Symmetric>::qmax - QuantSpec<NBits, Symmetric>::qmin);
    static constexpr int zero = -Min;
    static_assert(zero >= QuantSpec<NBits, Symmetric>::qmin && zero <= QuantSpec<NBits, Symmetric>::qmax, "Zero must be between QMin and QMax");
};
template <typename QSpec>
concept QuantSpecType = requires {
    typename QSpec;
    QSpec::qmin;
    QSpec::qmax;
    QSpec::scale;
    QSpec::zero;
};

template <QuantSpecType QSpec>
struct QuantizedNumber {
    int value;

    QuantizedNumber(int v) : value(v) {
        assert(value >= QSpec::qmin && value <= QSpec::qmax && "Value out of quantization range");
    }

    QuantizedNumber(float f) {
        int q = static_cast<int>(std::round(f / QSpec::scale) + QSpec::zero);
        if (q < QSpec::qmin) q = QSpec::qmin;
        if (q > QSpec::qmax) q = QSpec::qmax;
        value = q;
    }

    // TODO addition changes the range, so should return a wider quant type 
    QuantizedNumber<QSpec> operator+(const QuantizedNumber<QSpec>& other) const {
        return QuantizedNumber<QSpec>(this->value + other.value);
    }

    // TODO should be computed only once and saved
    float dequantize() const {
        return (value - QSpec::zero) * QSpec::scale;
    }

    //template <typename QSpecNew>
    //QuantizedNumber<QSpecNew> requantize1() {
    //    return QuantizedNumber<QSpecNew>(this->dequantize());
    //}
    // copy ctor is requantization
    // TODO have a variant where we dont dequantize, ie create the new QuantizedNumber using pure scale manipulations
    template <typename QSpecOld>
    QuantizedNumber(const QuantizedNumber<QSpecOld>& other) : QuantizedNumber(other.dequantize()) {}

    // Make cout << possible
    friend std::ostream& operator<<(std::ostream& os, const QuantizedNumber& qn) {
        os << "QuantizedNumber(value=" << qn.value
           << ", dequantized=" << qn.dequantize()
           << ", scale=" << QSpec::scale
           << ", zero=" << QSpec::zero
           << ", qmin=" << QSpec::qmin
           << ", qmax=" << QSpec::qmax;
        // Try to print Min/Max if available
        if constexpr (requires { QSpec::Min; QSpec::Max; }) {
            os << ", min=" << QSpec::Min << ", max=" << QSpec::Max;
        }
        os << ")";
        return os;
    }
};

int main() {
    using Q1 = QuantSpecScaleZero<4.0f, 0, 8, true>;
    static_assert(Q1::scale == 4.0f);
    static_assert(Q1::zero == 0);
    std::cout << "Q1 scale: " << Q1::scale << ", zero: " << Q1::zero << "\n";

    using Q2 = QuantSpecMinMax<-1.0f, 0.0f, 8, false>;
    static_assert(Q2::scale == 0.00392156886f);
    static_assert(Q2::zero == 1);
    std::cout << "Q2 scale: " << Q2::scale << ", zero: " << Q2::zero << "\n";


    QuantizedNumber<Q1> n0_i_q1_sz(10);
    QuantizedNumber<Q2> n1_i_q2_mm(20);
    // auto c = a + b; compile time type failure
    QuantizedNumber<Q1> n2_i_q1_sz(20);
    auto n3 = n0_i_q1_sz + n2_i_q1_sz;
    QuantizedNumber<Q2> n3_f_q1_sz(0.4f);
    auto n4 = n1_i_q2_mm + n3_f_q1_sz;


    QuantizedNumber<Q2> n4_f_q1_sz(4);
    

    // copy ctor from different quant spec, or requantization
    QuantizedNumber<Q1> n7_requant(n0_i_q1_sz);
    auto n8 = n7_requant + n0_i_q1_sz;
    std::cout << "n7: " << n8 << std::endl;

    return 0;
}