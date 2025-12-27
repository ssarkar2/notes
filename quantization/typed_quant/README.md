
# A note on types

Quantized quantities have an inherent scale/zero associated with them, and essentially different scale/zeroes are different datatypes. 


## Motivation
If we try to add 2 quantized quantities with differing ranges, we should get a compile time error. Encoding quantization parameters as types helps catch errors early and ensures that only compatible quantized values are combined, reducing bugs in numerical code that relies on quantization.


Note this would be much easier to do as runtime checks, but this note/code is more of an exercise in encoding info in compile type types/checks.

## Specs
1. Should be able to define quant spec as `scale`/`zero` or `Min`/`Max`
2. Types of the same range should add, else it should be a compile time error

## Try 1


### Simplification
1. `Min`/`Max` are defined as ints (though practically they are floats)


### Design
TL;DR: canonicalization to scale/zero followed by storing the scale/zero as type level metadata (template params or static members).

1. We need a way to canonilize scale/zero and min/max representation of quantization.
    - Specify a canonical representation: `QuantSpec<Scale, Zero>` 
    - Create a type `Canonicalized<T>` that expects `T` to contain static values `scale`/`zero`, which it then forwards to `QuantSpec<T::Scale, T::Zero>`
    - The non canonical spec is `MinMaxSpec<Min, Max, QMin, QMax>`, which computes `scale` and `zero` internally at compile time
2. A struct `Quantized<QS>` containing a `value` and its associated `QuantSpec`. Instantiating `Quantized` with different `QuantSpec`s will give different types
3. Some helpers for constructing `Quantized` types: `QuantFromScaleZero<Scale, Zero>`, `QuantFromMinMax<Min, Max>`
4. The `SameQuant` concept, that allows us to bound `operator+`'s inputs
5. Use `ratio` (rational numbers) to represent scale. `Min`/`Max` are kept as int, but could be represented as rationals as well. `std::ratio` works well with template types unline floats.




### Compile and run

```bash
g++ -std=c++20 quant_types.cpp
./a.out
```


## Try 2

### Design


1. A base class `QuantSpec<int NBits, bool Symmetric>`
2. That is subclassed into `QuantSpecScaleZero<float Scale, int Zero, int NBits, bool Symmetric>` and `QuantSpecMinMax<float Min, float Max, int NBits, bool Symmetric>`
3. At compile time, some static asserts perform some sanity checks, and everything is canonicalised to scale/zero representation, and concept `QuantSpecType` is satisfied
4. `QuantizedNumber<QSpec>` can be constructed using an int, or a float (in which case its quantized), or from another `QuantizedNumber` (in which case it undergoes requantization by undergoing dequant and quant).
5. addition works only on `QuantizedNumber`s with same `QSpec`, as expected of the type system



### Compile and run

```bash
g++ -std=c++20 quant_types_2.cpp
./a.out
```


### TODOS
1. addn should return a wider quant type 
2. when adding dont dequant, just adjust scales
3. code isnt well tested



