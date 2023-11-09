using lab3;
using MathNet.Numerics;

IEnumerable<double> coefficients = new[] {-25.7283, -35.3942, 6.0951, 1};
double a = -10;
double b = 10;
Polynomial y = new Polynomial(coefficients);

var intervals = NonlinearEquations.GetRootsIntervals(y, a, b);
Console.WriteLine(NonlinearEquations.Bisection(y, intervals[0]));
