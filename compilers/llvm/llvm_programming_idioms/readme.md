# Conventions and guidelines
Some notes from [this](https://llvm.org/docs/ProgrammersManual.html). Most of these are bespoke solutions to things "normal" C++ does, but using templates so we have compile time computations/guarantees, so things are faster

## Table of Contents
1. [Static polymorphism](#static-polymorphism)
2. [RTTI](#RTTI)
3. [Errors handling, exceptions and expected](#errors-handling,-exceptions-and-expected)
4. [Degugging](#Debugging)
5. [Data structures](##Data-structures)

## Static polymorphism
LLVM uses a lot of CRTP to define base->derived relationships (rather than the usual subclassing)for faster execution. See this for some [background](../cpp_background/mixin_crtp/readme.md). In this document see the [recoverable error handling](#Recoverable) section for an example where CRTP is used to create subclasses

## RTTI
LLVM rolls out its own RTII (instead of `dynamic_cast`) in `llvm/Support/Casting.h`. See this [section](https://llvm.org/docs/ProgrammersManual.html#isa). For example


### Using kinds and classof
Classes can opt in by following [this](https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html)

1. Changes in base class
    1. public
        1. `enum BaseClassKind` to enumerate the types of children class it might have
        2. `BaseClassKind getKind() const { return Kind; }`
        3. In the constructor, accepta  kind to set private variable `Kind`.
    2. private: `const BaseClassKind Kind;`
2. Changes in child class:
    1. in the ctor, call Base ctor with the appropriate `Kind`
    2. Add a static `classof` function `static bool classof(const Shape *S) {return S->getKind() == SK_Square;}`. `classof` accepts a ancestor pointer.
3. Now we can use: `Shape *S = ...; if (isa<Circle>(S)) {...}`


```c++
bool b = isa<Constant>(V);
...
if (auto *AI = dyn_cast<AllocationInst>(Val)) {
  // ...
}
```

### For open class heirarchies
Use CRTP with `RTTIExtends`.
```c++
class Shape : public RTTIExtends<Shape, RTTIRoot>

class Square : public RTTIExtends<Square, Shape> 

class Circle : public RTTIExtends<Circle, Shape>
```



In the next section we will see how error handling does not use regular exceptions, because of similar performance reasons

## Errors handling, exceptions and expected
Some highlights from [here](https://llvm.org/docs/ProgrammersManual.html#error-handling).

LLVM has 2 categories of errors: programmatic and recoverable.

### Programmatic: assert + llvm_unreachable 

Asserts can be used to express invariants
`assert(isPhysReg(R) && "All virt regs should have been allocated already.");`

`llvm_unreachable` is used to mark places where control flow should never reach
```c++
enum { Foo, Bar, Baz } X = foo();

switch (X) {
  case Foo: /* Handle Foo */; break;
  case Bar: /* Handle Bar */; break;
  default:
    llvm_unreachable("X should be Foo or Bar here");
}
```

### Recoverable
For example: missing file, malformed inputs etc. These errors need to propagate up and handled or error-reported to user using `Error` class (a wrapper for user defined errors, allowing arbitrary information to be attached to describe the error).

1. Class structure: 
    1. `Error` is the opposite of `Optional`/`Expected`. The empty case is success, the non-empty case contains error information
    2. Custom error classes are constructed using CRTP on templated base class `ErrorInfo`
3. Propagation:
    1. Errors can be converted to bool
    2. Errors cannot be dropped without checking for success. Even if it returns a success, we will get a runtime error. It can be moved up the stack for someone else to check.
    3. Multiple errors can be joined into a list by `joinErrors`
4. Handling: Use `handleErrors` or `handleAllErrors`. See code sample below
    1. `handleAllErrors` is expected to handle all errors, so it will abort if it cannot
    2. `handleErrors`: if it returns an `Error`, so its output must be checked.

See [this](https://llvm.org/docs/ProgrammersManual.html#recoverable-errors) for details

```c++
// A custom error class
class BadFileFormat : public ErrorInfo<BadFileFormat> {
public:
  static char ID;
  std::string Path;

  BadFileFormat(StringRef Path) : Path(Path.str()) {}

  void log(raw_ostream &OS) const override {
    OS << Path << " is malformed";
  }

  std::error_code convertToErrorCode() const override {
    return make_error_code(object_error::parse_failed);
  }
};
```

Propagating errors
```c++
Error foo(StringRef Path) {
  if (!check(Path)){
    return make_error<BadFileFormat>(Path);
  }
  // ... use path
  return Error::success();
}

...
// implicit conversion to boolean. True if error
if (auto E = foo(path))
    return E;
```

Handling errors
```c++
     auto E = foo(<...>); // <- foo returns failure with BadFileFormat.

     // handleErrors can take variadic list of handlers
     auto NewE =
     handleErrors(std::move(E),
       [](const BadFileFormat &M) {
         // Deal with the error.
       },
       [](std::unique_ptr<OtherError> M) -> Error {
         if (canHandle(*M)) {
           // handle error.
           return Error::success();
         }
         // Couldn't handle this error instance. Pass it up the stack.
         return Error(std::move(M));
     });

     // if (NewE)...
     // may need to check 
```

### Last resort
`report_fatal_error` calls installed error handlers, prints a message and aborts the program.

For reporting errors to the user (errors which are not expected to be fixed) use `StringError`.
```c++
make_error<StringError>("Bad executable", errc::executable_format_error);
createStringError(errc::executable_format_error, "Bad executable");
// or inconvertibleErrorCode() if there is no errc error, or you know it will not be converted
```

### Expected
Similar to `std::expected<T,E>`. Implicitly converts to bool (true for success)

```c++
Expected<FormattedFile> openFormattedFile(StringRef Path) {
  if (auto Err = checkFormat(Path))
    return std::move(Err); // Error return
  return FormattedFile(Path); // Expected value
}

Error processFormattedFile(StringRef Path) {
  // Try to open a formatted file
  if (auto FileOrErr = openFormattedFile(Path)) {
    auto &File = *FileOrErr; // success path
    ...
  } else
    // On error, extract the Error value and return it.
    return FileOrErr.takeError();
}


Error processFormattedFileAlternate(StringRef Path) {
  // Try to open a formatted file
  auto FileOrErr = openFormattedFile(Path);
  // can return success or error. error converts to True
  if (auto Err = FileOrErr.takeError())
    // On error, extract the Error value and return it.
    return Err;
  // On success, grab a reference to the file and continue.
  auto &File = *FileOrErr;
  ...
}
```


## Debugging
### Logs
```c++
LLVM_DEBUG(dbgs() << "I am here!\n");
```

Enabled with `-debug` option.

For specific prints use `-debug-only`
```c++
#define DEBUG_TYPE "foo"
LLVM_DEBUG(dbgs() << "'foo' debug type\n");
#undef  DEBUG_TYPE
DEBUG_WITH_TYPE("bar", dbgs() << "'bar' debug type\n");
```

```bash
opt < a.bc > /dev/null -mypass -debug-only=foo
```

### Statistics

```c++
#define DEBUG_TYPE "mypassname"   // This goes after any #includes.
STATISTIC(NumXForms, "The # of times I did stuff");
```

Increment the variable in code where we want to count something
```
++NumXForms;
```

### Run certain number of times
```c++
DEBUG_COUNTER(DeleteAnInstruction, "passname-delete-instruction",
              "Controls which instructions get delete");
...
if (DebugCounter::shouldExecute(DeleteAnInstruction))
  I->eraseFromParent();
```

```bash
$ opt --debug-counter=passname-delete-instruction=2-3 -passname
```

This will skip the above code the first two times we hit it, then execute it 2 times, then skip the rest of the executions.

### Graph views
Functions such as:
1. `Function::viewCFG()` (with instructions)
2. `Function::viewCFGOnly()` (without instructions)
3. `DAG.viewGraph()`




## Data structures
LLVM has its own set of [data structures](https://llvm.org/docs/ProgrammersManual.html#picking-the-right-data-structure-for-a-task) which might provide more efficient replacements for STL in some cases

### Ref classes

LLVM has some lightweight non-owning wrappers.

#### StringRef / Twine
Similar to `std::string_view`
1. non owning
2. non mutable
3. efficient (eg prevent copies when moving around)

StringRef has some extra functionalities over string_view, like case insensitive comparision, numeric comparisions, edit distance etc

#### function_ref
1. provides type erasure
2. Non-owner
3. Efficient (no copies)

It is more lightweight that std::function.

#### Arrays
1. ArrayRef
2. MutableArrayRef
3. SmallVectorImpl


### Small classes
Such as `SmallString`, `SmallVector`, `SmallSet`, `SmallPtrSet`, `SmallBitVector` etc




