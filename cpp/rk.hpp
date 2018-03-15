#ifndef FILE_RK
#define FILE_RK

#include <solve.hpp>
using namespace ngsolve;
#include <python_ngstd.hpp>



template <class T, class APP>
class ExplicitRK {

  /* Explicit RK methods have C[0] = 0 and lower triangular A:
       
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


  Matrix<T> _K;          // Storage for intermediate quantities
  
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
	   {0.0,  0.0,   1.0}};
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

  inline APP &
  FromApplication() { return static_cast<APP &>(*this); }

  inline void
  F(double z, BareVector<T> Y) { return FromApplication().F(z, Y); }
  
  inline void
  SolveIVP(Vector<T> & Y, double t, double h, int numsteps)  { 

    // With initial value at time t given by input Y = Y(t),
    // solve IVP and output solution at time t + numsteps * h.
    
    _h = h;
    for (int i=0; i<numsteps; i++) 
      Step(t+i*_h, Y);
  }

  inline void
  SolveIVP(Matrix<T> & Y, double t0, double h, int numsteps)  { 

    // With initial value at time t0 given by input Y.Row(0),
    // solve IVP and output solution in remaining rows of the
    // matrix Y such that  Y.Row(i) = solution at time t + i * h.
    
    _h = h;
    for (int i=0; i<numsteps-1; i++) {
      Y.Row(i+1) = Y.Row(i);
      Step(t0+i*h, Y.Row(i+1));
    }
  }

  void inline 
  Step(double t, Vector<T> & Y)  {

    /* Implement this without creating new memory:

        ynew = y + h ∑ᵢ bᵢ kᵢ,     where 

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



extern bool RunRKTests(); 

#endif // FILE_RK
