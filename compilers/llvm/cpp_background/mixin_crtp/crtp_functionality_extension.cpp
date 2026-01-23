#include <iostream>

template <typename D>
class Equal {
    public:
    bool operator==(const Equal<D>& rhs) const {
        return static_cast<D const* >(this)->get() == static_cast<D const* >(&rhs)->get();
    }
};

template <typename D>
class Print {
    public:
    void print() const {
        std::cout << "---->" << static_cast<D const* >(this)->get() << std::endl;
    }
};

class OneDPoint : public Equal<OneDPoint>, public Print<OneDPoint> {
    public:
    OneDPoint(double x) : m_x(x) {}
    double get() const {return m_x;}

    private:
    double m_x;
};

class KeyValuePair : public Equal<KeyValuePair>, public Print<KeyValuePair> {
    public:
    KeyValuePair(int k, int v) : m_k(k), m_v(v) {}
    int get() const {return m_k;}

    private:
    int m_k;
    int m_v;
};

template <typename D>
void helper(D a0, D a1, D a2) {
    std::cout << (a0 == a1) << " " << (a0 == a2) << std::endl;
    a0.print(); a1.print(); a2.print();
    Equal<D>* e0 = &a0; Print<D>* p0 = &a0;
    Equal<D>* e1 = &a1; Print<D>* p1 = &a1;
    Equal<D>* e2 = &a2; Print<D>* p2 = &a2;
    std::cout << (*e0 == *e1) << " " << (*e0 == *e2) << std::endl;
    p0->print(); p1->print(); p2->print();
}

int main() {
    OneDPoint odp0(2.0);
    OneDPoint odp1(2.0);
    OneDPoint odp2(3.0);
    helper<OneDPoint>(odp0, odp1, odp2);

    KeyValuePair kv0(1,2);
    KeyValuePair kv1(1,3);
    KeyValuePair kv2(2,3);
    helper<KeyValuePair>(kv0, kv1, kv2);
}
