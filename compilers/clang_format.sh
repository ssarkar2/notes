find . -name *.h -not -path "*/build/*" -o -name *.cpp -not -path "*/build/*" -o -name *.cc -not -path "*/build/*" -o -name *.hpp -not -path "*/build/*" | xargs clang-format -i -style=Google

