class MyClass<int val, int N = 10> {
  int value = val;
  int size = N;
  int v = value;
}

// Define a record that uses the default values
def CustomRecord0 : MyClass<42> { } // (42, 10, 42)

def CustomRecord1 : MyClass<32, 20> { } // (32, 20, 32)
def CustomRecord2 : MyClass<32, 20> { 
    let value = 10; // override 32->10 in the body
} // (10,20,10)

def CustomRecord4 : MyClass<?,?> { } // (?,?,value)


// 2 styles of let
def : MyClass<32, 20> {  // anonymous, name provided by tablegen
    let v = 10; // override 32->10 in the body
} // (32,20,10)

// Outer "defaults" with inner overrides
let size = 20, value = 1 in {
  let v = 50 in {
    def foo : MyClass<32, 20> {} // (1,20,50)
    def bar : MyClass<42, 30> { let value = 100; } // (100,20,50)
  }
}



