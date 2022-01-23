double* getVector(int N);

double** getIdentity(int N);

void printMatrix(double** matrix, int N, int M);

void printVector(double* vector, int N);

typedef double* (multiply_t)(double**, double*, const long);

void runSanityCheck(multiply_t);