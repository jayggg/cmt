#ifndef FILE_RK
#define FILE_RK

#include <solve.hpp>
using namespace ngsolve;
#include <python_ngstd.hpp>

#include <iostream>
#include <cmath>
#include <cfloat>


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

     To follow this notation, we will set C and B as Vectors, 
     while A is set to a Table (with first row length 0).
     An s-stage Runge-Kutta method does the following:

          Y <- Y + h (B₀ K₀ + B₁ K₁ + ... + Bₛ₋₁ Kₛ₋₁),
       
       where Kᵢ's are vectors defined by 

       K₀  = F(t, Y)
       K₁  = F(t + C₁ h, Y + h A₁₀ K₀)
       K₂  = F(t + C₂ h, Y + h (A₂₀ K₀ + A₂₁ K₁))
       :
       :
       :
       i.e., we have 

         Kᵢ = F(t + Cᵢ h, Y + h (Aᵢ₀ K₀ + Aᵢ₁ K₁ + ... + Aᵢ,ᵢ₋₁ Kᵢ₋₁)),
       
       for i = 0, 1, ..., s-1.

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
    Array<int> rowsizes(_s);
    for (int i=0; i<_s; i++) rowsizes[i]=i;
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

      _C = new Vector<>({0.,    1./2., 1./2., 1.   });
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

    for (int r=1; r <_s; r++)
      for (int c=0; c < rowsizes[r]; c++) 
	(*_A)[r][c] = A[r-1][c];
  }

  inline APP &
  FromApplication() { return static_cast<APP &>(*this); }

  inline void
  F(double z, FlatVector<T> Y) { return FromApplication().F(z, Y); }
  
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
  Step(double t, FlatVector<T> Y)  {

    /* Implement this without creating new memory:

          Y <- Y + h (B₀ K₀ + B₁ K₁ + ... + Bₛ₋₁ Kₛ₋₁),
       
       where Kᵢ's are vectors defined by 

       K₀  = F(t, Y)
       K₁  = F(t + C₁ h, Y + h A₁₀ K₀)
       K₂  = F(t + C₂ h, Y + h (A₂₀ K₀ + A₂₁ K₁))
       :
       :
       
       i.e., we have 

         Kᵢ = F(t + Cᵢ h, Y + h (Aᵢ₀ K₀ + Aᵢ₁ K₁ + ... + Aᵢ,ᵢ₋₁ Kᵢ₋₁)),
       
       for i = 0, 1, ..., s-1.

     */

    for (int i=0; i<_s; i++) {  // K₀, K₁, ..., Kₛ₋₁ will be made here
      
      // _Kᵢ <– Y
      _K.Row(i) = Y;
      
      for (int j=0; j<i; j++) 
	// Kᵢ <-  h * (Aᵢ₀ K₀ + Aᵢ₁ K₁ + ... + Aᵢ,ᵢ₋₁ Kᵢ₋₁)
	_K.Row(i) += _h * (*_A)[i][j] * _K.Row(j);

      // Kᵢ <-  F(t + Cᵢ h, Kᵢ)
      F(t + (*_C)[i]*_h, _K.Row(i));
    }

    // Y <-  Y + h ∑ᵢ Bᵢ Kᵢ
    for (int i=0; i<_s; i++) 
      Y += _h * (*_B)[i] * _K.Row(i);
  }
  
  void Print() const {
    cout << "Explicit RK method of " << _s
	 << " stages with the following Butcher tableau:" << endl;
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
