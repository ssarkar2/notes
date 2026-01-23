#include <iostream>


class OneDPoint {
    public:
    OneDPoint(double x) : m_x(x) {}
    double get() const {return m_x;}

    private:
    double m_x;
};

class KeyValuePair {
    public:
    KeyValuePair(int k, int v) : m_k(k), m_v(v) {}
    int get() const {return m_k;}

    private:
    int m_k;
    int m_v;
};

template<typename T>
struct Print : T
{
    public:
    template<typename... Args>
    Print(Args&&... args) : T(std::forward<Args>(args)...) {}
    void print() const {
        std::cout << "---->" << this->get() << std::endl;
    }
};


template<typename T>
struct Equal : T
{
    public:
    template<typename... Args>
    Equal(Args&&... args) : T(std::forward<Args>(args)...) {}
    bool operator==(const Equal<T>& rhs) const {
        return this->get() == rhs.get();
    }
};

int main() {
    Print<OneDPoint> pod(1.0);
    pod.print();

    Print<KeyValuePair> pkvp(2, 2);
    pkvp.print();

    Equal<Print<OneDPoint>> pod1(3.0);
    pod1.print();
    Equal<Print<OneDPoint>> pod2(3.0);
    std::cout << (pod1 == pod2) << "\n";

    Print<Equal<OneDPoint>> pod3(3.0);
    pod3.print();
    Print<Equal<OneDPoint>> pod4(4.0);
    std::cout << (pod3 == pod4) << "\n";
    return 0;
}