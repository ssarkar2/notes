#include <iostream>
#include <chrono>
#include <cmath>

class Base {
    public:
    virtual ~Base() {}
    virtual int compute() = 0;
};

class Derived : public Base {
    public:
    Derived(int x) : m_x(x) {}

    int compute () override {
        m_x += 0.1;
        auto y = std::sin(m_x) + std::cos(m_x);
        return y;
    }
    
    private:
    int m_x;
};


template<typename T>
class BaseT {
    public:
    int compute(){
        return static_cast<T*>(this)->compute();
    }
    // private:
    // Base() {}
    // friend DerivedT
};

class DerivedT : public BaseT<DerivedT> {
    public:
    DerivedT(int x) : m_x(x) {}

    int compute() {
        m_x += 0.1;
        auto y = std::sin(m_x) + std::cos(m_x);
        return y;
    }
    private:
    double m_x;
};


template <typename D, typename B>
void time_me() {
    auto start = std::chrono::high_resolution_clock::now();
    D d(2);
    B* p = &d;
    int x = 0;
    for (int i = 0; i < 10000000; i++) {
        x = p->compute();
    }
    std::cout << "final res = " << x << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "\nExecution time: " << duration.count() << " milliseconds" << std::endl;
}

int main() {
    time_me<Derived, Base>();
    time_me<DerivedT, BaseT<DerivedT>>();

    // Can be called thru derived object or through base class ptr
    DerivedT d(2);
    d.compute();
    BaseT<DerivedT>* p = &d;
    p->compute();
    return 0;
}