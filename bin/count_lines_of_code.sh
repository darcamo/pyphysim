# This requires the cloc program to be installed.
cd ..
echo xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
echo Lines of codes in Pyphysim library
cloc pyphysim --exclude-lang="Bourne Shell,C"

echo
echo xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
echo Lines of codes in simulators
cloc apps --exclude-lang="Bourne Shell"

echo
echo xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
echo Lines of code in test files
cloc tests --exclude-lang="Bourne Shell"
