# LLVM pattern matching
Source code [here](https://llvm.org/doxygen/PatternMatch_8h_source.html) in `PatternMatch.h`

Here is a sample LLVM code matching  `(x && c1) || (y && c2)`

```
Value *Exp = ...
Value *X, *Y;  ConstantInt *C1, *C2;      // (X & C1) | (Y & C2)
if (match(Exp, m_Or(m_And(m_Value(X), m_ConstantInt(C1)),
                      m_And(m_Value(Y), m_ConstantInt(C2))))) {
    ... Pattern is matched and variables are bound ...
}
```


## Top match function
Lets look at the outermost `match`. This is simply a call `pattern.match(value)`
```
template <typename Val, typename Pattern> bool match(Val *V, const Pattern &P) {
  return const_cast<Pattern &>(P).match(V);
}
```

## Patterns
Any class that implements a `match` function. It can be basic classes or higher level combiner classes like `match_combine_or` (which checks if it matches atleast 1 of the 2 patterns). Because everyone implements `bool match(...)` any struct can recursively keep calling deeper matches till one of them returns a boolean

### Single vs groups
There are many functions beginnig with `m_` that return pattern matchers for certain ops (like `m_Or`) or patterns (eg `m_Signum`). There might be specific versions like `m_Shl`/`m_Shr` for shift-left or shift-right individually or a combined one like `m_Shift`.

### Predicated patterns
Some patterns (like `m_Shift`) subclass from a `Predicate` class (like `is_shift_op`). Predicate classes implement `isOpType`



## Deep Dive: Check if its an "Add" and capture its inputs


### Caller code
Consider the example:
```
Value *X, *Y;
// Create a matcher for the addition operation
auto AddMatcher = m_BinOp(m_Value(X), m_Value(Y), Instruction::Add);

// Use the match function to check if V matches the pattern
if (match(V, AddMatcher)) {
    // Pattern matched, X and Y are bound to the operands of the addition
    errs() << "Matched addition: " << *X << " + " << *Y << "\n";
} else {
    errs() << "No match found.\n";
}
```

### Relevant code from PatternMatch.h
```
inline bind_ty<Value> m_Value(Value *&V) { return V; }

template <typename LHS_t, typename RHS_t, bool Commutable = false>
struct SpecificBinaryOp_match
    : public BinaryOp_match<LHS_t, RHS_t, 0, Commutable> {
  unsigned Opcode;
 
  SpecificBinaryOp_match(unsigned Opcode, const LHS_t &LHS, const RHS_t &RHS)
      : BinaryOp_match<LHS_t, RHS_t, 0, Commutable>(LHS, RHS), Opcode(Opcode) {}
 
  template <typename OpTy> bool match(OpTy *V) {
    return BinaryOp_match<LHS_t, RHS_t, 0, Commutable>::match(Opcode, V);
  }
};
 
/// Matches a specific opcode.
template <typename LHS, typename RHS>
inline SpecificBinaryOp_match<LHS, RHS> m_BinOp(unsigned Opcode, const LHS &L,
                                                const RHS &R) {
  return SpecificBinaryOp_match<LHS, RHS>(Opcode, L, R);
}
```

### Walkthrough
So we start at: `match(V, AddMatcher)` and gradually expand:

`AddMatcher.match(V)`

`m_BinOp(m_Value(X), m_Value(Y), Instruction::Add).match(V)`


`SpecificBinaryOp_match<class_match<Value>, class_match<Value>>(Instruction::Add, m_Value(X), m_Value(Y)).match(V)`

`BinaryOp_match<class_match<Value>, class_match<Value>, 0, Commutable>::match(Instruction::Add, V)`

Now replacing in the match function:
```
Opc --> Instruction::Add
L --> m_Value(X)
R --> m_Value(Y)

if (V->getValueID() == Value::InstructionVal + Opc) {  // "V" must be Instruction::Add
      auto *I = cast<BinaryOperator>(V);
      // in this example this part just captures inputs to add for furthur use
      return (L.match(I->getOperand(0)) && R.match(I->getOperand(1))) ||
             (Commutable && L.match(I->getOperand(1)) && // swap and compare
              R.match(I->getOperand(0)));
    }
    return false;
```

Lets dig into `L.match(I->getOperand(0))`:

`m_Value(X).match(I->getOperand(0))`

`bind_ty<Value>(X).match(I->getOperand(0))`


`bind_ty` code:

```
template <typename Class> struct bind_ty {
  Class *&VR;
 
  bind_ty(Class *&V) : VR(V) {}
 
  template <typename ITy> bool match(ITy *V) { // V is I->getOperand(0)
    if (auto *CV = dyn_cast<Class>(V)) {  // Class is Value
      VR = CV;                            // X stores I->getOperand(0) now
      return true;
    }
    return false;
  }
};
```

Thus `bind_ty` at construction time had stored a pointer to `Value` and if it successfully casts `V` to a `Value` it assigns it to `X`. Thus after the pattern matching `X` ends up with the input to the addition. Same with `Y`.


