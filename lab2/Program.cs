using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

var source = new double[3, 3] { { 4, -1, 0 }, { -1, 4, -1 }, { 0, -1, 3 } };

var sourceC = new double[5, 5] { { 0.01, 0, -0.02, 0, 0 },
                                 { 0.01, 0.01, -0.02, 0, 0 },
                                 { 0, 0.01, 0.01, 0, -0.02 },
                                 { 0, 0, 0.01, 0.01, 0 },
                                 { 0, 0, 0, 0.01, 0.01 } };

var sourceD = new double[5, 5] { { 1.33, 0.21, 0.17, 0.12, -0.13 },
                                 { -0.13, -1.33, 0.11, 0.17, 0.12 },
                                 { 0.12, -0.13, -1.33, 0.11, 0.11 },
                                 { 0.17, 0.12, -0.13, -1.33, 0.11 },
                                 { 0.11, 0.67, 0.12, -0.13, -1.33 } };

var sourceB = new double[5] { 1.2, 2.2, 4.0, 0.0, -1.2 };

var C = Matrix<double>.Build.DenseOfArray(sourceC);
var D = Matrix<double>.Build.DenseOfArray(sourceD);

var A = 12 * C + D;
var b = Vector<double>.Build.DenseOfArray(sourceB);

Console.WriteLine("Исходные данные:");
Console.WriteLine(A);
Console.WriteLine(b);
Console.WriteLine();
Console.WriteLine("Решение системы с помощью математического пакета: ");
Console.WriteLine(A.Solve(b));
Console.WriteLine();

if (checkDominant(A))
{
    Console.WriteLine("Нет диагональной доминированности");
    return;
}

const double error = 1e-4;
var ans = iterations(A, b, error);
//var ans = zeidel(A, b, error);


Console.WriteLine(ans);


static Vector<double> iterations(Matrix<double> A, Vector<double> b, double err)
{
    var A_modified = Matrix<double>.Build.DenseOfMatrix(A);
    var b_modified = Vector<double>.Build.DenseOfVector(b);

    for (var i = 0; i < A.RowCount; i++)
        for (var j = 0; j < A.ColumnCount; j++)
            A_modified[i, j] /= -A[i, i];

    for (var i = 0; i < b.Count; i++)
        b_modified[i] = b[i] / A[i, i];

    double acc = 1;
    var prev = Vector<double>.Build.Dense(b.Count);
    var curr = Vector<double>.Build.Dense(b.Count);

    int iterations = 0;

    while (acc > err)
    {
        curr = A_modified.Multiply(prev) + b_modified + prev;
        acc = 0;
        for (var i = 0; i < curr.Count; i++)
        {
            if (Math.Abs(curr[i] - prev[i]) > acc)
            {
                acc = Math.Abs(curr[i] - prev[i]);
                break;
            }
        }
        prev = curr;
        iterations++;
    }

    Console.WriteLine($"В итерационном методе кол-во итераций: {iterations}");

    return prev;
}

static Vector<double> zeidel(Matrix<double> A, Vector<double> b, double err)
{
    double acc = 1;
    var ans = Vector<double>.Build.Dense(b.Count);
    var problem = Vector<double>.Build.Dense(b.Count);

    int iterations = 0;

    while (acc > err)
    {
        for (int j = 0; j < A.RowCount; j++)
        {
            double diff = b[j];
            for (int k = 0; k < A.RowCount; k++)
                diff -= A[j, k] * ans[k];
            diff /= A[j, j];
            diff += ans[j];
            problem[j] = Math.Abs(diff - ans[j]);
            ans[j] = diff;
            acc = Math.Abs(problem.Max());
        }
        iterations++;
    }

    Console.WriteLine($"В методе Зейделя кол-во итераций: {iterations}");

    return ans;
}

static bool checkDominant(Matrix<double> A)
{
    for (int i = 0; i < A.RowCount; i++)
    {
        double rowSum = 0;
        for (int j = 0; j  < A.ColumnCount; j++)
            rowSum += A[i, j];
        rowSum -= A[i, i];
        if (A[i, i] < rowSum)
            return false;
    }

    return true;
}