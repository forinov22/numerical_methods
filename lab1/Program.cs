using MathNet.Numerics.LinearAlgebra;

var sourceC = new [,]
{
    { 0.2, 0, 0.2, 0, 0 },
    { 0, 0.2, 0, 0.2, 0 },
    { 0.2, 0, 0.2, 0, 0.2 },
    { 0, 0.2, 0, 0.2, 0 },
    { 0, 0, 0.2, 0, 0.2 }
};
// ReSharper disable once InconsistentNaming
var C = Matrix<double>.Build.DenseOfArray(sourceC);
var sourceD = new [,]
{
    { 2.33, 0.81, 0.67, 0.92, -0.53 },
    { -0.53, 2.33, 0.81, 0.67, 0.92 },
    { 0.92, -0.53, 2.33, 0.81, 0.67 },
    { 0.67, 0.92, -0.53, 2.33, 0.81 },
    { 0.81, 0.67, 0.92, -0.53, 2.33 }
};
// ReSharper disable once InconsistentNaming
var D = Matrix<double>.Build.DenseOfArray(sourceD);
var sourceB = new [] { 4.3, 4.1, 4.3, 4.1, 4.3 };
// ReSharper disable once InconsistentNaming
var A = C.Multiply(12).Add(D);
var b = Vector<double>.Build.DenseOfArray(sourceB);


Console.WriteLine("Source matrx A:");
Console.WriteLine(A);
Console.WriteLine("Condition number:");
Console.WriteLine(A.ConditionNumber());
Console.WriteLine();
Console.WriteLine("Source matrix b:");
Console.WriteLine(b);
Console.WriteLine();
Console.WriteLine("Solution of packet Math.Net.Numerics");
Console.WriteLine(A.Solve(b));
Console.WriteLine();
Console.WriteLine("Solution of gauss method:");
gauss(ref A, ref b);
Console.WriteLine();
Console.WriteLine("Solution of gaussColumn method:");
gaussColumn(ref A, ref b);
Console.WriteLine();
Console.WriteLine("Solution of gaussMax method:");
gaussMax(ref A, ref b);

static (int xIndex, int yIndex) findMaxIndexesAfterRow(ref Matrix<double> A, int i)
{
    var max = A[i, i];
    var xIndex = i;
    var yIndex = i;
    for (var k = i; k < A.RowCount; k++)
    {
        for (var j = i; j < A.ColumnCount; j++)
        {
            if (A[k, j] > max)
            {
                max = A[k, j];
                xIndex = k;
                yIndex = j;
            }
        }
    }
    return (xIndex, yIndex);
}

// ReSharper disable once InconsistentNaming
static int maxElementInColumnAfterRow(ref Matrix<double> A, int i)
{
    var max = A[i, i];
    var row = i;
    for (var k = i; k < A.RowCount; k++)
    {
        if (A[k, i] > max)
        {
            max = A[k, i];
            row = k;
        }
    }
    return row;
}

// ReSharper disable once InconsistentNaming
static void swapRows(ref Matrix<double> A, int i, int j)
{
    var row1 = A.Row(i);
    var row2 = A.Row(j);
    A.SetRow(i, row2);
    A.SetRow(j, row1);
}

// ReSharper disable once InconsistentNaming
static void swapColumns(ref Matrix<double> A, int i, int j)
{
    var column1 = A.Column(i);
    var column2 = A.Column(j);
    A.SetColumn(i, column2);
    A.SetColumn(j, column1);
}

// ReSharper disable once InconsistentNaming
static bool checkDiagonal(ref Matrix<double> A)
{
    for (var i = 0; i < A.RowCount; i++)
    {
        if (A[i, i] == 0)
        {
            var check = true;
            for (var j = 0; j < A.ColumnCount; j++)
            {
                if (A[i, j] != 0 && A[j, i] != 0)
                {
                    swapColumns(ref A, i, j);
                    check = false;
                    break;
                }
            }
            if (check)
                return false;
        }
    }
    return true;
}

// ReSharper disable once InconsistentNaming
static void run(ref Matrix<double> A, ref Vector<double> b, int i)
{
    for (var j = i; j < A.RowCount; j++)
    {
        var factor = A[j, i - 1] / A[i - 1, i - 1];
        var rowToSubstract = A.Row(i - 1) * factor;
        b[j] -= factor * b[i - 1];
        A.SetRow(j, A.Row(j) - rowToSubstract);
    }
}

// ReSharper disable once InconsistentNaming
static void straightRun(ref Matrix<double> A, ref Vector<double> b)
{
    for (var i = 1; i < A.RowCount; i++)
        run(ref A, ref b, i);
}

static void straightRunColumn(ref Matrix<double> A, ref Vector<double> b)
{
    for (var i = 1; i < A.RowCount; i++)
    {
        var index = maxElementInColumnAfterRow(ref A, i - 1);
        swapRows(ref A, i - 1, index);
        (b[i - 1], b[index]) = (b[index], b[i - 1]);
        run(ref A, ref b, i);
    }
}

// ReSharper disable once InconsistentNaming
static int[] straightRunMax(ref Matrix<double> A, ref Vector<double> b)
{
    var positions = new int[b.Count];
    for (var i = 0; i < b.Count; i++)
        positions[i] = i;
    
    for (var i = 1; i < A.RowCount; i++)
    {
        var (maxIndexX, maxIndexY) = findMaxIndexesAfterRow(ref A, i - 1);
        swapColumns(ref A, maxIndexY, i);
        (positions[i], positions[maxIndexY]) = (positions[maxIndexY], positions[i]);
        swapRows(ref A, maxIndexX, i);
        (b[i], b[maxIndexX]) = (b[maxIndexX], b[i]);
        run(ref A, ref b, i);
    }
    return positions;
}

// ReSharper disable once InconsistentNaming
static Vector<double> reverseRun(ref Matrix<double> A, ref Vector<double> b)
{
    var ans = Vector<double>.Build.Dense(A.RowCount);
    for (var i = A.RowCount - 1; i >= 0; i--)
    {
        double diff = 0;
        for (var j = A.ColumnCount - 1; j > i; j--)
            diff += ans[j] * A[i, j];
        ans[i] = (b[i] - diff) / A[i, i];
    }
    return ans;
}

// ReSharper disable once InconsistentNaming
static Vector<double> residual(ref Matrix<double> A, ref Vector<double> b, ref Vector<double> x) => A.Multiply(x) - b;

static double error(Vector<double> residual) => residual.L2Norm();

// ReSharper disable once InconsistentNaming
static void gauss(ref Matrix<double> A, ref Vector<double> b)
{
    var aCopy = Matrix<double>.Build.Dense(A.RowCount, A.ColumnCount);
    A.CopyTo(aCopy);
    var bCopy = Vector<double>.Build.Dense(b.Count);
    b.CopyTo(bCopy);
    var check = checkDiagonal(ref aCopy);
    if (!check)
    {
        Console.WriteLine("Бесконечное количество решений или нет решений!");
        return;
    }
    straightRun(ref aCopy, ref bCopy);
    var ans = reverseRun(ref aCopy, ref bCopy);
    Console.WriteLine(ans);
    Console.WriteLine($"Error: {error(residual(ref A, ref b, ref ans))}");
}

// ReSharper disable once InconsistentNaming
static void gaussColumn(ref Matrix<double> A, ref Vector<double> b)
{
    var aCopy = Matrix<double>.Build.Dense(A.RowCount, A.ColumnCount);
    A.CopyTo(aCopy);
    var bCopy = Vector<double>.Build.Dense(b.Count);
    b.CopyTo(bCopy);
    var check = checkDiagonal(ref aCopy);
    if (!check)
    {
        Console.WriteLine("Бесконечное количество решений или нет решений!");
        return;
    }
    straightRunColumn(ref aCopy, ref bCopy);
    var ans = reverseRun(ref aCopy, ref bCopy);
    Console.WriteLine(ans);
    Console.WriteLine($"Error: {error(residual(ref A, ref b, ref ans))}");
}

// ReSharper disable once InconsistentNaming
static void gaussMax(ref Matrix<double> A, ref Vector<double> b)
{
    var aCopy = Matrix<double>.Build.Dense(A.RowCount, A.ColumnCount);
    A.CopyTo(aCopy);
    var bCopy = Vector<double>.Build.Dense(b.Count);
    b.CopyTo(bCopy);
    var check = checkDiagonal(ref aCopy);
    if (!check)
    {
        Console.WriteLine("Бесконечное количество решений или нет решений!");
        return;
    }
    var positions = straightRunMax(ref aCopy, ref bCopy);
    var ans = reverseRun(ref aCopy, ref bCopy);
    var ansRebuild = Vector<double>.Build.DenseOfVector(ans);
    for (var i = 0; i < ans.Count; i++)
    {
        ansRebuild[positions[i]] = ans[i];
    }
    Console.WriteLine(ansRebuild);
    Console.WriteLine($"Error: {error(residual(ref A, ref b, ref ansRebuild))}");
}
