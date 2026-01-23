# Visitor

## Motivation
Consider an `Animal` heirarchy. We have multiple animals, aand they have things they can do, say speaking and walking. When designing the class heirarchy we could add virtual functions `speak()` and `walk()`. However we might want to extend the capabilities of the animals on the fly later on.

The visitor pattern discussed here is one way to do it. You can also check out the [CRTP/mixin](../mixin_crtp/readme.md) section to see how to do it at compile time.


## Example
See [visitor.cpp](visitor.cpp).

1. We have a visitor heirarchy, with the base visitor declaring the following. This class declares the interface `on_visiting`, and overloads it for all teh animals we might encounter.
```cpp
virtual void on_visiting(DogV* p) = 0;
virtual void on_visiting(CatV* p) = 0;
```
2. Each subclass of the base `Visitor` provides a new "skill" by implementing `on_visiting` for each animnal


## Usage in LLVM
We could have an AST program representation, and visitors that traverse the tree doing stuff like printing, generating IR etc. See examples [here](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl03.html) and [here](https://github.com/PacktPublishing/Learn-LLVM-17/blob/main/Chapter02/calc/src/CodeGen.cpp#L10)
