#include "rk.hpp"

// Make a simple application class for testing

template <class T>
class App: public ExplicitRK<T, App<T>> {
  function<void(double, FlatVector<T>)> _f;
public:
  App(function<void(double, FlatVector<T>)> ff, int s, int n)
    :  ExplicitRK<T, App<T>>(s, n) { _f = ff; }
  void F(double z, FlatVector<T> Y) { return _f(z, Y); }
};


// Test function

bool
RunRKTests() {

  bool success = true;
  double tol = 1.e-9;   // this is our machine epsilon for tests
  double h = 0.01;
  int nsteps = 1000;
  double t = 0.0;
  double tf = t + nsteps * h;
  Vector<int> stages({1,2,4,7});
  
  cout << endl << "* Checking scalar case:" << endl;
  Vector<function<void(double, FlatVector<double>)>>
    fun({
        [](double t, FlatVector<double> Y){ Y[0]=1.0;},
        [](double t, FlatVector<double> Y){ Y[0]=t;},
        [](double t, FlatVector<double> Y){ Y[0]=t*t*t;},
        [](double t, FlatVector<double> Y){ Y[0]=t*t*t*t;}
      });
  Vector<> exact({1+tf, 1+0.5*tf*tf, 1+(tf*tf*tf*tf/4.0), 1+(tf*tf*tf*tf*tf/5.0)});
  Vector<> y({1.0});

  
  for (int i=0; i<stages.Size(); i++) {
    y[0] = 1.0;
    cout << "** Checking stage " << stages[i] << " RK" << endl;

    App<double> a(fun[i], stages[i], 1);
    a.SolveIVP(y, t, h, nsteps);
    if (abs(y[0] - exact[i]) < tol)  
      cout <<"    Passed.  diff = " << y[0] - exact[i] << endl;	 
    else {
      success = false;
      cout <<"    *** FAILED!" << endl;
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
  exactv.Row(3) =  v + (tf*tf*tf*tf*tf/5.0)*ones;
  
  Vector<function<void(double, FlatVector<double>)>>
    vfn({ [&](double t, FlatVector<double> Y) {for (int i=0; i<n; i++) Y[i]=1.0;},
          [&](double t, FlatVector<double> Y) {for (int i=0; i<n; i++) Y[i]=t;},
          [&](double t, FlatVector<double> Y) {for (int i=0; i<n; i++) Y[i]=t*t*t;},
          [&](double t, FlatVector<double> Y) {for (int i=0; i<n; i++) Y[i]=t*t*t*t;}
      });
  
  for (int i=0; i<stages.Size(); i++) {
    v = {1.0, 0.0, 0.0, 1.0, 0.5};
    cout << "** Checking stage " << stages[i] << " RK for vector" << endl;

    App<double> av(vfn[i], stages[i], n);    
    av.SolveIVP(v, t, h, nsteps);
    
    if (L2Norm(v - exactv.Row(i)) < tol)
      cout <<"    Passed.  diff = " << L2Norm(v - exactv.Row(i))  << endl;
    else {
      success = false;
      cout <<"    *** FAILED!  diff = " << L2Norm(v - exactv.Row(i))  << endl;
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
  exactvc.Row(3) =  vc + (tf*tf*tf*tf*tf/5.0)*ones;
  
  Vector<function<void(double, FlatVector<Complex>)>>
    vcfn({[&](double t, FlatVector<Complex> Y) {for (int i=0; i<n; i++) Y[i]=1.0;},
          [&](double t, FlatVector<Complex> Y) {for (int i=0; i<n; i++) Y[i]=t;},
          [&](double t, FlatVector<Complex> Y) {for (int i=0; i<n; i++) Y[i]=t*t*t;},
          [&](double t, FlatVector<Complex> Y) {for (int i=0; i<n; i++) Y[i]=t*t*t*t;}
      });
  
  for (int i=0; i<stages.Size(); i++) {
    vc = {1.0, 0.0, Complex(0, 3.0), 1.0, Complex(0.5, 1)};

    cout << "** Checking stage " << stages[i] << " RK for complex vector" << endl;

    App<Complex> avc(vcfn[i], stages[i], n);    
    avc.SolveIVP(vc, t, h, nsteps);

    if (L2Norm(vc - exactvc.Row(i)) < tol)
      cout <<"    Passed.  diff = " << L2Norm(vc - exactvc.Row(i))  << endl;
    else {
      success = false;
      cout <<"    *** FAILED!  diff = " << L2Norm(vc - exactvc.Row(i))  << endl;
      cout <<"!!! vc = " <<  vc << endl
           <<"!!! exact = " << exactvc.Row(i) << endl;
    }
  }

  return success;

}



