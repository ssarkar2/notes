
# Setup


Following [this](https://github.com/abenkhadra/llvm-pass-tutorial)
which is an extention of [this](https://www.cs.cornell.edu/~asampson/blog/llvm.html) + [this](https://github.com/sampsyo/llvm-pass-skeleton )



We can have in-source vs out-of-source passes. In-source pass lets us control compilation process (such as building in debug mode)


## Building from source

From the [releases](https://releases.llvm.org/) page of llvm, we go to [1.19.1](https://github.com/llvm/llvm-project/releases/tag/llvmorg-19.1.1) downloads page. From here we get the [source](https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-19.1.1.zip) zip

### Get the source

```bash
cd /mnt/c/sayantan/llvm
wget https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-19.1.1.zip
sudo apt-get install unzip
unzip llvmorg-19.1.1.zip
export LLVM_SRC=/mnt/c/sayantan/llvm/llvm-project-llvmorg-19.1.1
```

### Build tools
Skippable if `cmake --version` shows >3.20

Upgrade cmake to atleast 3.20. Using [this](https://medium.com/@yulin_li/how-to-update-cmake-on-ubuntu-9602521deecb)
```bash
sudo apt purge --autoremove cmake
sudo make uninstall
sudo apt update
sudo apt install -y build-essential libssl-dev


cd /mnt/c/sayantan/llvm/cmake
wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz
cd cmake-3.20.0
sudo ./bootstrap
# hits error on WSL


```

Trying a different path
```bash
sudo apt-get install apt-transport-https

sudo apt update
# failed: E: The repository 'https://apt.kitware.com/ubuntu focal InRelease' is not signed.

sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 16FAAD7AF99A65E2


sudo apt update
# works now

sudo apt install cmake # installs 3.31.6
```


### Setup build dir

```bash
mkdir /mnt/c/sayantan/llvm/llvm-build
cd /mnt/c/sayantan/llvm/llvm-build
cmake -DCMAKE_BUILD_TYPE=Debug -DLLVM_TARGETS_TO_BUILD=X86 /mnt/c/sayantan/llvm/llvm-project-llvmorg-19.1.1/llvm



 cmake --build . # -j 4


 # [ 82%] Linking CXX executable ../../bin/llvm-c-test
# /usr/bin/ld: final link failed: Input/output error
# collect2: error: ld returned 1 exit status
#make[2]: *** [tools/llvm-c-test/CMakeFiles/llvm-c-test.dir/build.make:341: bin/llvm-c-test] Error 1
#make[1]: *** [CMakeFiles/Makefile2:25023: tools/llvm-c-test/CMakeFiles/llvm-c-test.dir/all] Error 2
#make: *** [Makefile:156: all] Error 2

# above error was due to out-of-disk-space. made space, reran above cmd, passed.

# by default it installs in /usr/local, but lets specify a separate dir
export LLVM_HOME=/mnt/c/sayantan/llvm/llvm-home-1911
mkdir $LLVM_HOME
cmake -DCMAKE_INSTALL_PREFIX=$LLVM_HOME -P cmake_install.cmake
```

Verify by running something like: `$LLVM_HOME/bin/llc --version`.


### Install clang

```

```
