/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
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
 
#define nrn_init _nrn_init__Kaxon
#define _nrn_initial _nrn_initial__Kaxon
#define nrn_cur _nrn_cur__Kaxon
#define _nrn_current _nrn_current__Kaxon
#define nrn_jacob _nrn_jacob__Kaxon
#define nrn_state _nrn_state__Kaxon
#define _net_receive _net_receive__Kaxon 
#define _f_evaluate_fct _f_evaluate_fct__Kaxon 
#define evaluate_fct evaluate_fct__Kaxon 
#define states states__Kaxon 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define gkaxon _p[0]
#define gkaxon_columnindex 0
#define ik _p[1]
#define ik_columnindex 1
#define n _p[2]
#define n_columnindex 2
#define ek _p[3]
#define ek_columnindex 3
#define Dn _p[4]
#define Dn_columnindex 4
#define _g _p[5]
#define _g_columnindex 5
#define _ion_ek	*_ppvar[0]._pval
#define _ion_ik	*_ppvar[1]._pval
#define _ion_dikdv	*_ppvar[2]._pval
 
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
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_Exp(void);
 static void _hoc_evaluate_fct(void);
 static void _hoc_states(void);
 static void _hoc_vtrap(void);
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

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_Kaxon", _hoc_setdata,
 "Exp_Kaxon", _hoc_Exp,
 "evaluate_fct_Kaxon", _hoc_evaluate_fct,
 "states_Kaxon", _hoc_states,
 "vtrap_Kaxon", _hoc_vtrap,
 0, 0
};
#define Exp Exp_Kaxon
#define vtrap vtrap_Kaxon
 extern double Exp( double );
 extern double vtrap( double , double );
 /* declare global and static user variables */
#define ntau ntau_Kaxon
 double ntau = 0;
#define nexp nexp_Kaxon
 double nexp = 0;
#define ninf ninf_Kaxon
 double ninf = 0;
#define usetable usetable_Kaxon
 double usetable = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_Kaxon", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "ntau_Kaxon", "ms",
 "gkaxon_Kaxon", "mho/cm2",
 "ik_Kaxon", "mA/cm2",
 0,0
};
 static double delta_t = 1;
 static double n0 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "ninf_Kaxon", &ninf_Kaxon,
 "nexp_Kaxon", &nexp_Kaxon,
 "ntau_Kaxon", &ntau_Kaxon,
 "usetable_Kaxon", &usetable_Kaxon,
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
 
static int _ode_count(int);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"Kaxon",
 "gkaxon_Kaxon",
 0,
 "ik_Kaxon",
 0,
 "n_Kaxon",
 0,
 0};
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 6, _prop);
 	/*initialize range parameters*/
 	gkaxon = 0.0319;
 	_prop->param = _p;
 	_prop->param_size = 6;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ek */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 
}
 static void _initlists();
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _Kaxon_reg() {
	int _vectorized = 0;
  _initlists();
 	ion_reg("k", -10000.);
 	_k_sym = hoc_lookup("k_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 6, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
 	hoc_register_cvode(_mechtype, _ode_count, 0, 0, 0);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Kaxon /mnt/c/Users/analk/OneDrive/Desktop/2021-12-20-ModellingPaper/Fig2/Turi_et_al_2018_o/mechanisms/Kaxon.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double *_t_ninf;
 static double *_t_nexp;
 static double *_t_ntau;
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int _f_evaluate_fct(double);
static int evaluate_fct(double);
static int states();
 static void _n_evaluate_fct(double);
 
static int  states (  ) {
   evaluate_fct ( _threadargscomma_ v ) ;
   n = n + nexp * ( ninf - n ) ;
   
/*VERBATIM*/
	return 0;
  return 0; }
 
static void _hoc_states(void) {
  double _r;
   _r = 1.;
 states (  );
 hoc_retpushx(_r);
}
 static double _mfac_evaluate_fct, _tmin_evaluate_fct;
 static void _check_evaluate_fct();
 static void _check_evaluate_fct() {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  static double _sav_dt;
  static double _sav_celsius;
  if (!usetable) {return;}
  if (_sav_dt != dt) { _maktable = 1;}
  if (_sav_celsius != celsius) { _maktable = 1;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_evaluate_fct =  - 200.0 ;
   _tmax =  100.0 ;
   _dx = (_tmax - _tmin_evaluate_fct)/300.; _mfac_evaluate_fct = 1./_dx;
   for (_i=0, _x=_tmin_evaluate_fct; _i < 301; _x += _dx, _i++) {
    _f_evaluate_fct(_x);
    _t_ninf[_i] = ninf;
    _t_nexp[_i] = nexp;
    _t_ntau[_i] = ntau;
   }
   _sav_dt = dt;
   _sav_celsius = celsius;
  }
 }

 static int evaluate_fct(double _lv){ _check_evaluate_fct();
 _n_evaluate_fct(_lv);
 return 0;
 }

 static void _n_evaluate_fct(double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_evaluate_fct(_lv); return; 
}
 _xi = _mfac_evaluate_fct * (_lv - _tmin_evaluate_fct);
 if (isnan(_xi)) {
  ninf = _xi;
  nexp = _xi;
  ntau = _xi;
  return;
 }
 if (_xi <= 0.) {
 ninf = _t_ninf[0];
 nexp = _t_nexp[0];
 ntau = _t_ntau[0];
 return; }
 if (_xi >= 300.) {
 ninf = _t_ninf[300];
 nexp = _t_nexp[300];
 ntau = _t_ntau[300];
 return; }
 _i = (int) _xi;
 _theta = _xi - (double)_i;
 ninf = _t_ninf[_i] + _theta*(_t_ninf[_i+1] - _t_ninf[_i]);
 nexp = _t_nexp[_i] + _theta*(_t_nexp[_i+1] - _t_nexp[_i]);
 ntau = _t_ntau[_i] + _theta*(_t_ntau[_i+1] - _t_ntau[_i]);
 }

 
static int  _f_evaluate_fct (  double _lv ) {
   double _lq10 , _ltinc , _lalpha , _lbeta ;
 _lq10 = 1.0 ;
   _ltinc = - dt * _lq10 ;
   _lalpha = 0.018 * vtrap ( _threadargscomma_ - ( _lv - 25.0 ) , 25.0 ) ;
   _lbeta = 0.0036 * vtrap ( _threadargscomma_ _lv - 35.0 , 12.0 ) ;
   ntau = 1.0 / ( _lalpha + _lbeta ) ;
   ninf = _lalpha * ntau ;
   nexp = 1.0 - Exp ( _threadargscomma_ _ltinc / ntau ) ;
    return 0; }
 
static void _hoc_evaluate_fct(void) {
  double _r;
    _r = 1.;
 evaluate_fct (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double vtrap (  double _lx , double _ly ) {
   double _lvtrap;
 if ( fabs ( _lx / _ly ) < 1e-6 ) {
     _lvtrap = _ly * ( 1.0 - _lx / _ly / 2.0 ) ;
     }
   else {
     _lvtrap = _lx / ( Exp ( _threadargscomma_ _lx / _ly ) - 1.0 ) ;
     }
   
return _lvtrap;
 }
 
static void _hoc_vtrap(void) {
  double _r;
   _r =  vtrap (  *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
double Exp (  double _lx ) {
   double _lExp;
 if ( _lx < - 100.0 ) {
     _lExp = 0.0 ;
     }
   else {
     _lExp = exp ( _lx ) ;
     }
   
return _lExp;
 }
 
static void _hoc_Exp(void) {
  double _r;
   _r =  Exp (  *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ hoc_execerror("Kaxon", "cannot be used with CVODE"); return 0;}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_k_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 2, 4);
 }

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  n = n0;
 {
   n = ninf ;
   }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
  ek = _ion_ek;
 initmodel();
 }}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   ik = gkaxon * n * n * n * n * ( v - ek ) ;
   }
 _current += ik;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
  ek = _ion_ek;
 _g = _nrn_current(_v + .001);
 	{ double _dik;
  _dik = ik;
 _rhs = _nrn_current(_v);
  _ion_dikdv += (_dik - ik)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ik += ik ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
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
 
}}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  ek = _ion_ek;
 { error =  states();
 if(error){fprintf(stderr,"at line 39 in file Kaxon.mod:\n    SOLVE states\n"); nrn_complain(_p); abort_run(error);}
 } }}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
   _t_ninf = makevector(301*sizeof(double));
   _t_nexp = makevector(301*sizeof(double));
   _t_ntau = makevector(301*sizeof(double));
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/mnt/c/Users/analk/OneDrive/Desktop/2021-12-20-ModellingPaper/Fig2/Turi_et_al_2018_o/mechanisms/Kaxon.mod";
static const char* nmodl_file_text = 
  "COMMENT\n"
  "Potassium current for the axon\n"
  "ENDCOMMENT\n"
  "\n"
  "UNITS {\n"
  "    (mA) = (milliamp)\n"
  "    (mV) = (millivolt)\n"
  "}\n"
  " \n"
  "NEURON {\n"
  "    SUFFIX Kaxon\n"
  "    USEION k READ ek WRITE ik\n"
  "    RANGE gkaxon, ik\n"
  "    GLOBAL ninf, nexp, ntau\n"
  "}\n"
  " \n"
  "INDEPENDENT { t FROM 0 TO 1 WITH 1 (ms) }\n"
  " \n"
  "PARAMETER {\n"
  "    v               (mV)\n"
  "    celsius = 24    (degC)\n"
  "    dt              (ms)\n"
  "    gkaxon  = .0319 (mho/cm2)\n"
  "    ek      = -90   (mV)\n"
  "}\n"
  " \n"
  "STATE { n }\n"
  " \n"
  "ASSIGNED {\n"
  "    ik (mA/cm2)\n"
  "    ninf \n"
  "	nexp \n"
  "	ntau (ms)\n"
  "}\n"
  " \n"
  "INITIAL { n = ninf }\n"
  "\n"
  "BREAKPOINT {\n"
  "    SOLVE states\n"
  "	ik = gkaxon*n*n*n*n*(v - ek)    \n"
  "}\n"
  "\n"
  "PROCEDURE states() {	:exact when v held constant\n"
  "	evaluate_fct(v)\n"
  "	n = n + nexp*(ninf - n)\n"
  "	VERBATIM\n"
  "	return 0;\n"
  "	ENDVERBATIM \n"
  "}\n"
  "UNITSOFF\n"
  "\n"
  "PROCEDURE evaluate_fct(v(mV)) {  \n"
  "	:Computes rate and other constants at current v.\n"
  "	:Call once from HOC to initialize inf at resting v.\n"
  "\n"
  "	LOCAL q10, tinc, alpha, beta\n"
  "	TABLE ninf, nexp, ntau DEPEND dt, celsius FROM -200 TO 100 WITH 300\n"
  "	\n"
  "	:q10 = 3^((celsius - 24)/10)\n"
  "	q10   = 1	: BPG\n"
  "	tinc  = -dt*q10\n"
  "	alpha = 0.018*vtrap(-(v-25),25)\n"
  "	beta  = 0.0036*vtrap(v-35,12)\n"
  "	ntau  = 1/(alpha + beta)\n"
  "	ninf  = alpha*ntau\n"
  "	nexp  = 1-Exp(tinc/ntau)\n"
  "}\n"
  "\n"
  "FUNCTION vtrap(x,y) {\n"
  "	:Traps for 0 in denominator of rate eqns.\n"
  "	if (fabs(x/y) < 1e-6) {\n"
  "		vtrap = y*(1 - x/y/2)\n"
  "	}else{\n"
  "		vtrap = x/(Exp(x/y) - 1)\n"
  "	}\n"
  "}\n"
  "\n"
  "FUNCTION Exp(x) {\n"
  "	if (x < -100) {\n"
  "		Exp = 0\n"
  "	}else{\n"
  "		Exp = exp(x)\n"
  "	}\n"
  "}\n"
  "UNITSON\n"
  ;
#endif
