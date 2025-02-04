/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__ANSynapse
#define _nrn_initial _nrn_initial__ANSynapse
#define nrn_cur _nrn_cur__ANSynapse
#define _nrn_current _nrn_current__ANSynapse
#define nrn_jacob _nrn_jacob__ANSynapse
#define nrn_state _nrn_state__ANSynapse
#define _net_receive _net_receive__ANSynapse 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define onset _p[0]
#define onset_columnindex 0
#define tau1 _p[1]
#define tau1_columnindex 1
#define tau2 _p[2]
#define tau2_columnindex 2
#define Nfrac _p[3]
#define Nfrac_columnindex 3
#define Ntau1 _p[4]
#define Ntau1_columnindex 4
#define Ntau2 _p[5]
#define Ntau2_columnindex 5
#define eta _p[6]
#define eta_columnindex 6
#define Mg _p[7]
#define Mg_columnindex 7
#define gamma _p[8]
#define gamma_columnindex 8
#define gmax _p[9]
#define gmax_columnindex 9
#define e _p[10]
#define e_columnindex 10
#define i _p[11]
#define i_columnindex 11
#define g _p[12]
#define g_columnindex 12
#define gA _p[13]
#define gA_columnindex 13
#define gN _p[14]
#define gN_columnindex 14
#define Agmax _p[15]
#define Agmax_columnindex 15
#define Ngmax _p[16]
#define Ngmax_columnindex 16
#define v _p[17]
#define v_columnindex 17
#define _g _p[18]
#define _g_columnindex 18
#define _nd_area  *_ppvar[0]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_duale(void*);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 "duale", _hoc_duale,
 0, 0
};
#define duale duale_ANSynapse
 extern double duale( _threadargsprotocomma_ double , double );
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "Ntau2", 0.001, 1e+06,
 "Ntau1", 0.001, 1e+06,
 "gmax", 0, 1e+09,
 "tau2", 0.001, 1e+06,
 "tau1", 0.001, 1e+06,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "onset", "ms",
 "tau1", "ms",
 "tau2", "ms",
 "Ntau1", "ms",
 "Ntau2", "ms",
 "eta", "/mM",
 "Mg", "mM",
 "gamma", "/mV",
 "gmax", "umho",
 "e", "mV",
 "i", "nA",
 "g", "umho",
 "gA", "umho",
 "gN", "umho",
 0,0
};
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"ANSynapse",
 "onset",
 "tau1",
 "tau2",
 "Nfrac",
 "Ntau1",
 "Ntau2",
 "eta",
 "Mg",
 "gamma",
 "gmax",
 "e",
 0,
 "i",
 "g",
 "gA",
 "gN",
 0,
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 19, _prop);
 	/*initialize range parameters*/
 	onset = 0;
 	tau1 = 0.2;
 	tau2 = 2;
 	Nfrac = 0.5;
 	Ntau1 = 0.66;
 	Ntau2 = 80;
 	eta = 0.33;
 	Mg = 1;
 	gamma = 0.06;
 	gmax = 0;
 	e = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 19;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 2, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _ANsyn_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 19, 2);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 ANSynapse /mnt/c/Users/analk/OneDrive/Desktop/2021-12-20-ModellingPaper/Fig2/Turi_et_al_2018_o/mechanisms/ANsyn.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
double duale ( _threadargsprotocomma_ double _lx , double _ly ) {
   double _lduale;
 if ( _lx < 0.0  || _ly < 0.0 ) {
     _lduale = 0.0 ;
     }
   else {
     _lduale = exp ( - _lx ) - exp ( - _ly ) ;
     }
   
return _lduale;
 }
 
static double _hoc_duale(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  duale ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) );
 return(_r);
}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
 {
   Agmax = ( 1.0 - Nfrac ) * gmax ;
   Ngmax = Nfrac * gmax ;
   }

}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   gA = Agmax * ( ( tau1 * tau2 ) / ( tau1 - tau2 ) ) * duale ( _threadargscomma_ ( t - onset ) / tau1 , ( t - onset ) / tau2 ) ;
   gN = Ngmax * ( ( Ntau1 * Ntau2 ) / ( Ntau1 - Ntau2 ) ) * duale ( _threadargscomma_ ( t - onset ) / Ntau1 , ( t - onset ) / Ntau2 ) ;
   gN = gN / ( 1.0 + ( eta * Mg * exp ( - gamma * v ) ) ) ;
   g = gA + gN ;
   i = g * ( v - e ) ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/mnt/c/Users/analk/OneDrive/Desktop/2021-12-20-ModellingPaper/Fig2/Turi_et_al_2018_o/mechanisms/ANsyn.mod";
static const char* nmodl_file_text = 
  "COMMENT\n"
  "A synaptic current with two dual exponential function conductances,\n"
  "representing non-voltage-dependent AMPA and voltage-dependent NMDA\n"
  "components.  The basic dual exponential conductance is given by:\n"
  "         g = 0 for t < onset and\n"
  "         g = gmax*((tau1*tau2)/(tau1-tau2)) * (exp(-(t-onset)/tau1)-exp(-(t-onset)/tau2))\n"
  "         for t > onset (tau1 and tau2 are fast and slow time constants)\n"
  "The synaptic current is:\n"
  "        i = (gA + gN) * (v - e)      i(nanoamps), g(micromhos);\n"
  "NMDA model taken from Mel, J. Neurophys. 70:1086-1101, 1993\n"
  "BPG 1-12-00\n"
  "ENDCOMMENT\n"
  "                           \n"
  "INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}\n"
  "\n"
  "NEURON {\n"
  "    POINT_PROCESS ANSynapse\n"
  "    RANGE onset, gmax, e, i, g, gA, gN, tau1, tau2, Ntau1, Ntau2, eta, Mg, gamma, Nfrac\n"
  "    NONSPECIFIC_CURRENT i\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "    (nA) = (nanoamp)\n"
  "    (mV) = (millivolt)\n"
  "    (umho) = (micromho)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    onset=0 (ms)\n"
  "    tau1=.2 (ms)    <1e-3,1e6>\n"
  "    tau2=2 (ms)    <1e-3,1e6>\n"
  "    Nfrac=0.5\n"
  "    Ntau1=.66 (ms)    <1e-3,1e6>\n"
  "    Ntau2=80 (ms)    <1e-3,1e6>\n"
  "    eta=0.33 (/mM)\n"
  "    Mg=1 (mM)\n"
  "    gamma=0.06 (/mV)\n"
  "    gmax=0  (umho)  <0,1e9>\n"
  "    e=0 (mV)\n"
  "    v   (mV)\n"
  "}\n"
  "\n"
  "ASSIGNED { i (nA)  g (umho) gA (umho) gN (umho) Agmax (umho) Ngmax (umho)}\n"
  "\n"
  "INITIAL {\n"
  "    Agmax = (1-Nfrac)*gmax\n"
  "    Ngmax = Nfrac*gmax\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "    gA = Agmax*((tau1*tau2)/(tau1-tau2))*duale((t-onset)/tau1,(t-onset)/tau2)\n"
  "    gN = Ngmax*((Ntau1*Ntau2)/(Ntau1-Ntau2))*duale((t-onset)/Ntau1,(t-onset)/Ntau2)\n"
  "    gN = gN / (1 + (eta*Mg*exp(-gamma*v)))\n"
  "    g = gA + gN\n"
  "    i = g*(v - e)\n"
  "}\n"
  "\n"
  "FUNCTION duale(x,y) {\n"
  "    if (x < 0 || y < 0) {\n"
  "        duale = 0\n"
  "    }else{\n"
  "        duale = exp(-x) - exp(-y)\n"
  "    }\n"
  "}\n"
  ;
#endif
