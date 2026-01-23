#include <iostream>

struct Animal {
    virtual void speak() = 0;
    virtual void walk() = 0;
};

struct Dog : public Animal {
    virtual void speak() override {
        std::cout << "woof\n";
    }
    virtual void walk() override {
        std::cout << "lets go\n";
    }
};

struct Cat : public Animal {
    virtual void speak() override {
        std::cout << "meow\n";
    }
    virtual void walk() override {
        std::cout << "I'd rather stay on the sofa\n";
    }
};






struct AnimalV;
struct DogV;
struct CatV;

struct Visitor {
    virtual void on_visiting(DogV* p) = 0;
    virtual void on_visiting(CatV* p) = 0;
};

struct SpeakerVisitor : public Visitor {
    virtual void on_visiting(DogV* p) override {
        std::cout << "woof\n";
    }
    virtual void on_visiting(CatV* p) override {
        std::cout << "meow\n";
    }
};


struct WalkerVisitor : public Visitor {
    virtual void on_visiting(DogV* p) override {
        std::cout << "lets go\n";
    }
    virtual void on_visiting(CatV* p) override {
        std::cout << "I'd rather stay on the sofa\n";
    }
};

struct AnimalV {
    virtual void accept(Visitor* v) = 0;
};

struct DogV : public AnimalV {
    virtual void accept(Visitor* v) override {
        v->on_visiting(this);
    }
};

struct CatV : public AnimalV {
    virtual void accept(Visitor* v) override {
        v->on_visiting(this);
    }
};





int main() {
    Animal *a0 = new Dog();
    Animal *a1 = new Cat();
    a0->speak();
    a1->speak();
    a0->walk();
    a1->walk();


    AnimalV *av0 = new DogV();
    AnimalV *av1 = new CatV();
    Visitor* s = new SpeakerVisitor();
    av0->accept(s);
    av1->accept(s);

    Visitor* w = new WalkerVisitor();
    av0->accept(w);
    av1->accept(w);
    return 0;
}