// See https://aka.ms/new-console-template for more information

using MathNet.Numerics.LinearAlgebra;

var sourceC = new[,]
{
    { 0.2, 0, 0.2, 0, 0 },
    { 0, 0.2, 0, 0.2, 0 },
    { 0.2, 0, 0.2, 0, 0.2 },
    { 0, 0.2, 0, 0.2, 0 },
    { 0, 0, 0.2, 0, 0.2 }
};

var sourceD = new[,]
{
  { 2.33, 0.81, 0.67, 0.92, -0.53 },
  { 0.81, 2.33, 0.81, 0.67, 0.92 },
  { 0.67, 0.81, 2.33, 0.81, 0.92 },
  { 0.92, 0.67, 0.81, 2.33, -0.53 },
  { -0.53, 0.92, 0.92, -0.53, 2.33 }
};

var c = Matrix<double>.Build.DenseOfArray(sourceC);
var d = Matrix<double>.Build.DenseOfArray(sourceD);
var a = 12 * c + d;

Console.WriteLine(a);
var (values, vectors) = Jacobi(a);
Console.WriteLine(values);
Console.WriteLine(vectors);

Tuple<Vector<double>, Matrix<double>> Jacobi(Matrix<double> matrix)
{
    if (!matrix.IsSymmetric())
        throw new ArgumentException("Source matrix is not symmetric");
    
    var n = matrix.RowCount;
    const double maxIterations = 1000;
    var eigenvectors = Matrix<double>.Build.DenseDiagonal(n, n, 1);
    
    for (var iteration = 0; iteration < maxIterations; iteration++)
    {
        double maxOfDiagonal = 0.0;
        int p = 0, q = 0;
        
        for (var i = 0; i < n - 1; i++)
        {
            for (var j = i + 1; j < n; j++)
            {
                if (Math.Abs(matrix[i, j]) > maxOfDiagonal)
                {
                    maxOfDiagonal = Math.Abs(matrix[i, j]);
                    p = i;
                    q = j;
                }
            }
        }
        
        if (maxOfDiagonal < 1e-8)
            break;
        
        double angle = 0.5 * Math.Atan(2 * matrix[p, q] / (matrix[p, p] - matrix[q, q]));
        double cos = Math.Cos(angle);
        double sin = Math.Sin(angle);
        
        var rotationMatrix = Matrix<double>.Build.Dense(n, n);
        for (var i  = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                if (i == j && i != p && i != q)
                    rotationMatrix[i, j] = 1.0;
                else if (i == p && j == p)
                    rotationMatrix[i, j] = cos;
                else if (i == p && j == q)
                    rotationMatrix[i, j] = -sin;
                else if (i == q && j == p)
                    rotationMatrix[i, j] = sin;
                else if (i == q && j == q)
                    rotationMatrix[i, j] = cos;
                else
                    rotationMatrix[i, j] = 0.0;
            }
        }
        
        matrix = rotationMatrix.Transpose().Multiply(matrix).Multiply(rotationMatrix);
        eigenvectors = eigenvectors.Multiply(rotationMatrix);
    }
    
    var eigenValues = matrix.Diagonal();
    return new (eigenValues, eigenvectors);
}
