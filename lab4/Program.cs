

using MathNet.Numerics.LinearAlgebra;

var x = new double[]{0.5, 0.2};

var ans = Newton(x);

foreach (var a in ans.Item1)
{
    Console.WriteLine(a);
}

Console.WriteLine($"Iterations: {ans.Item2}");

(double[],int) Iterations(double[] x)
{
    int iterations = 0;
    var eps = 1e-5;
    double error = 1;

    while (error > eps)
    {
        var x1 = x[0];
        var x2 = x[1];
        x = EvaluateFunctions(x);
        var x11 = x[0];
        var x21 = x[1];
        error = Math.Max(Math.Abs(x1 - x11), Math.Abs(x2 - x21));
        ++iterations;
    }

    return (x, iterations);
}

(double[],int) Newton(double[] x)
{
    int iterations = 0;
    var eps = 1e-5;
    double error = 1;

    while (error > eps)
    {
        var x1 = x[0];
        var x2 = x[1];
        x = EvaluatePhi(x);
        var x11 = x[0];
        var x21 = x[1];
        error = Math.Max(Math.Abs(x1 - x11), Math.Abs(x2 - x21));
        ++iterations;
    }

    return (x, iterations);
}

double[] EvaluateFunctions(double[] x)
{
    x[0] = Math.Tan(x[0] * x[1] + 0.3);
    x[1] = Math.Sqrt((1 - x[0] * x[0]) / 2);
    return x;
}

Func<double, double, double>[,] Jacobi()
{
    var f11 = (double x, double y) => y / Math.Pow(Math.Cos(x * y + 0.3), 2) - 1;
    var f12 = (double x, double y) => x / Math.Pow(Math.Cos(x * y + 0.3), 2);
    var f21 = (double x, double y) => 2 * x;
    var f22 = (double x, double y) => 4 * y;
    
    return new[,] { { f11, f12 }, { f21, f22 } };
}

Func<double, double, double>[] System()
{
    var f1 = (double x, double y) => Math.Tan(x * y + 0.3) - x;
    var f2 = (double x, double y) => Math.Pow(x, 2) + 2 * Math.Pow(y, 2) - 1;
    
    return new[] { f1, f2 };
}

double[] EvaluatePhi(double[] x)
{
    var system = System();
    var jacobi = Jacobi();
    var xVec = Vector<double>.Build.DenseOfArray(x);
    var bSource = new double[2,2];
    
    for (var i = 0; i < 2; i++)
        for (var j = 0; j < 2; j++)
            bSource[i, j] = jacobi[i, j](x[0], x[1]);
        
    var b = Matrix<double>.Build.DenseOfArray(bSource);
    var a = -1 * b.Inverse();
    
    var fSource = new double[2];
    for (var i = 0; i < 2; i++)
        fSource[i] = system[i](x[0], x[1]);
    var f = Vector<double>.Build.DenseOfArray(fSource);
    
    return (xVec + a.Multiply(f)).ToArray();
}
