# This requires the cloc program to be installed.
cd ..
cloc apps cell comm comp ia MATLAB plot subspace util --exclude-lang="Bourne Shell,C"

echo
echo xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
echo Lines of code in test files
cloc tests --exclude-lang="Bourne Shell"
