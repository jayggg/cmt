#ifndef FILE_RK
#define FILE_RK

#include <solve.hpp>
using namespace ngsolve;
#include <python_ngstd.hpp>



template <class T>
class ExplicitRK {

  /* Explicit RK method have C[0] = 0 and lower triangular A:
       
            0    |
	   C[1]  | A[1,0]
	   C[2]  | A[2,0]    A[2,1]
	    :    | :          
	    :    | :          
	  C[s-1] | A[s-1,0]  A[s-1,1]  ...  A[s-1, s-2]
	  ----------------------------------------------
	         | B[0]      B[1]      ...  B[s-1]

     Reference: https://en.wikipedia.org/wiki/Runge–Kutta_methods.
  */ 

protected:
  
  double _h;             // step size 
  int _s;                // number of stages
  int _n;                // number of equations in ODE system
  
  Vector<> * _B;         // Butcher tableau:     C  |  A  
  Vector<> * _C;	 //                      ---.----- 
  Table<double>  * _A;	 //                         |  B^T


  Matrix<T> _K;  // Intermediate 
  
public:
  
  ExplicitRK(int s, int n) : _K(s, n) {

    // Construct a pre-defined RK method with s stages
    
    _s = s;
    _h = 0.0;
    Array<int> rowsizes(_s-1);
    for (int i=0; i<_s-1; i++) rowsizes[i]=i+1;
    _A = new Table<double>(rowsizes);
    vector<vector<double>> A;
    
    switch(_s) {

    case 1: { // Explicit Euler

      /*  Butcher Tableau:	 
	 
    	  0 | 
    	  ------
    	    | 1
       */
      
      _C = new Vector<>({0.0});
      _B = new Vector<>({1.0});
      
      break;
    }
    case 2: { // Midpoint method
      
      /* Butcher Tableau:
	 
    	    0 | 
    	  1/2 | 1/2 
    	  -------------
    	      | 0    1
       */

      _C = new Vector<>({0.0, 0.5});
      _B = new Vector<>({0.0, 1.0});
      A = {{0.5}};
      break;
    }
    case 4: { // RK4 method

      /* Butcher Tableau:

	 0   |
	 1/2 |	1/2
	 1/2 |	0	1/2
	 1   |	0	0	1    
	 ----------------------------------
	     | 1/6	1/3	1/3	1/6

       */

      _C = new Vector<>({0.,    1./2., 1./2., 1    });
      _B = new Vector<>({1./6., 1./3., 1./3., 1./6.});
      A = {{0.5}, 		
	   {0.0,  0.5},   	
	   {0,    0,     1.}};
      break;
    }
      
    default:
      cout << "No " << _s << " stage RK case implemented!" << endl;
      break;
    }

    for (int r=0; r < _s-1; r++)
      for (int c=0; c < rowsizes[r]; c++) 
	(*_A)[r][c] = A[r][c];
  }

  void inline
  SolveIVP(const function<void(double, BareVector<T>)> & F,
	   Vector<T> & Y, double t, double h, int numsteps)  { 

    // With initial value at time t given by input Y = Y(t),
    // solve IVP and output solution at time t + numsteps * h.
    
    _h = h;
    for (int i=0; i<numsteps; i++) 
      Step(F, t+i*_h, Y);
  }

  void inline 
  Step(const function<void(double, BareVector<T>)> & F,
       double t, Vector<T> & Y)  {
    /* 
        ynew = y + h ∑ᵢ bᵢ kᵢ
        k₀ = f(t, y)
        k₁ = f(t + c₁ h, y + h a₁₀ k₀)
        k₂ = f(t + c₂ h, y + h (a₂₀ k₀ + a₂₁ k₁))
          : 
          :
	kᵢ = f(t + cᵢ h, y + h (aᵢ₀ k₀ + aᵢ₁ k₁ + ... + aᵢ,ᵢ₋₁ kᵢ₋₁))
     */

    for (int i=0; i<_s; i++) {

      // _kᵢ <– y
      _K.Row(i) = Y;

      for (int j=0; j<i; j++)
	// kᵢ <- h (aᵢ₀ k₀ + aᵢ₁ k₁ + ... + aᵢ,ᵢ₋₁ kᵢ₋₁)
	_K.Row(i) += _h * (*_A)[i][j] * _K.Row(j);

      // kᵢ <- F(t+cᵢh, kᵢ)
      F(t + (*_C)[i]*_h, _K.Row(i));
    }

    // y <-  y + h ∑ᵢ bᵢ kᵢ
    for (int i=0; i<_s; i++) 
      Y += _h * (*_B)[i] * _K.Row(i);
    
  }

    
  void Print() const {
    cout << "A:" << endl << *_A << endl;
    cout << "B:" << endl << *_B << endl;
    cout << "C:" << endl << *_C << endl;
  }
  
  ~ExplicitRK() {

    delete _A;
    delete _B;
    delete _C;
  }
  
};


bool
RunRKTests() {

  bool success = true;
  double h = 0.01;
  int nsteps = 1000;
  double t = 0.0;
  double tf = t + nsteps * h;

  Vector<int> stages({1,2,4});
  
  cout << endl << "* Checking scalar case:" << endl;
  Vector<function<void(double, BareVector<double>)>>
    fun({
	[](double t, BareVector<double> Y){ Y[0]=1.0;},
	  [](double t, BareVector<double> Y){ Y[0]=t;},
	    [](double t, BareVector<double> Y){ Y[0]=t*t*t;}
      });
  Vector<> exact({1+tf, 1+0.5*tf*tf, 1+(tf*tf*tf*tf/4.0)});
  Vector<> y({1.0});

  for (int i=0; i<stages.Size(); i++) {
    y[0] = 1.0;
    cout << "** Checking stage " << stages[i] << " RK" << endl;
    ExplicitRK<double> rk(stages[i], 1);
    rk.SolveIVP(fun[i], y, t, h, nsteps);
    if (abs(y[0] - exact[i]) < 1.e-10)
      cout <<"    Passed!" << endl;
    else {
      success = false;
      cout <<"    Failed!" << endl;
      cout <<"!!! y[0] = " <<  y[0] << " exact = " << exact[i]
	   << " diff = " << y[0] - exact[i] << endl;
    }
  }
  
  cout << endl << "* Checking vector case:" << endl;
  int n = 5;
  Vector<> v({1.0, 0.0, 0.0, 1.0, 0.5});
  Vector<> ones({1., 1., 1., 1., 1.});
  Matrix<double> exactv(stages.Size(),n);
  exactv.Row(0) =  v + tf*ones;
  exactv.Row(1) =  v + 0.5*tf*tf*ones;
  exactv.Row(2) =  v + (tf*tf*tf*tf/4.0)*ones;
  
  Vector<function<void(double, BareVector<double>)>>
    vfn({ [&](double t, BareVector<double> Y)  {for (int i=0; i<n; i++) Y[i]=1.0;},
	  [&](double t, BareVector<double> Y) {for (int i=0; i<n; i++) Y[i]=t;},
	  [&](double t, BareVector<double> Y) {for (int i=0; i<n; i++) Y[i]=t*t*t;}
      });
  
  for (int i=0; i<stages.Size(); i++) {
    v = {1.0, 0.0, 0.0, 1.0, 0.5};
    cout << "** Checking stage " << stages[i] << " RK for vector" << endl;
    ExplicitRK<double> rk1v(stages[i], n);
    rk1v.SolveIVP(vfn[i], v, t, h, nsteps); 
    if (L2Norm(v - exactv.Row(i)) < 1.e-10)
      cout <<"    Passed!" << endl;
    else {
      success = false;
      cout <<"    Failed!  diff = " << L2Norm(v - exactv.Row(i))  << endl;
      cout <<"!!! v = " <<  v << endl
	   <<"!!! exact = " << exactv.Row(i) << endl;
    }
  }

  cout << endl << "* Checking complex vector case:" << endl;
  Vector<Complex> vc(n);
  vc = {1.0, 0.0, Complex(0, 3.0), 1.0, Complex(0.5, 1)};
  Matrix<Complex> exactvc(stages.Size(),n);
  exactvc.Row(0) =  vc + tf*ones;
  exactvc.Row(1) =  vc + 0.5*tf*tf*ones;
  exactvc.Row(2) =  vc + (tf*tf*tf*tf/4.0)*ones;
  
  Vector<function<void(double, BareVector<Complex>)>>
    vcfn({[&](double t, BareVector<Complex> Y) {for (int i=0; i<n; i++) Y[i]=1.0;},
	  [&](double t, BareVector<Complex> Y) {for (int i=0; i<n; i++) Y[i]=t;},
	  [&](double t, BareVector<Complex> Y) {for (int i=0; i<n; i++) Y[i]=t*t*t;}
      });
  
  for (int i=0; i<stages.Size(); i++) {
    vc = {1.0, 0.0, Complex(0, 3.0), 1.0, Complex(0.5, 1)};
    cout << "** Checking stage " << stages[i]
	 << " RK for complex vector" << endl;
    ExplicitRK<Complex> rk1vc(stages[i], n);
    rk1vc.SolveIVP(vcfn[i], vc, t, h, nsteps); 
    if (L2Norm(vc - exactvc.Row(i)) < 1.e-10)
      cout <<"    Passed!" << endl;
    else {
            success = false;
      cout <<"    Failed!  diff = " << L2Norm(vc - exactvc.Row(i))  << endl;
      cout <<"!!! vc = " <<  vc << endl
	   <<"!!! exact = " << exactvc.Row(i) << endl;
    }
  }

  return success;

}



void
ExportExplicitRKComplex(py::module m) {

  py::class_<ExplicitRK<Complex>, shared_ptr<ExplicitRK<Complex>>>
    (m, "ExplicitRKComplex", "An explicit Runge-Kutta method")
    
    .def(py::init([](int s, int n) {
	  return shared_ptr<ExplicitRK<Complex>>(new
						 ExplicitRK<Complex>(s, n));
	}))
    
    .def("Print", &ExplicitRK<Complex>::Print)
    ;

}

void
ExportExplicitRK(py::module m) {

  py::class_<ExplicitRK<double>, shared_ptr<ExplicitRK<double>>>
    (m, "ExplicitRK", "An explicit Runge-Kutta method")
    
    .def(py::init([](int s, int n) {
	  return shared_ptr<ExplicitRK<double>>(new
						ExplicitRK<double>(s, n));
	}))
    
    .def("Print", &ExplicitRK<double>::Print)
    ;

}

#endif // FILE_RK
