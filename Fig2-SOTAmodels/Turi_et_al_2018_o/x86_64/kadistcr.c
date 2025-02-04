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
 
#define nrn_init _nrn_init__kadcr
#define _nrn_initial _nrn_initial__kadcr
#define nrn_cur _nrn_cur__kadcr
#define _nrn_current _nrn_current__kadcr
#define nrn_jacob _nrn_jacob__kadcr
#define nrn_state _nrn_state__kadcr
#define _net_receive _net_receive__kadcr 
#define rates rates__kadcr 
#define states states__kadcr 
 
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
#define gkabar _p[0]
#define gkabar_columnindex 0
#define ik _p[1]
#define ik_columnindex 1
#define gka _p[2]
#define gka_columnindex 2
#define n _p[3]
#define n_columnindex 3
#define l _p[4]
#define l_columnindex 4
#define ek _p[5]
#define ek_columnindex 5
#define ki _p[6]
#define ki_columnindex 6
#define ko _p[7]
#define ko_columnindex 7
#define Dn _p[8]
#define Dn_columnindex 8
#define Dl _p[9]
#define Dl_columnindex 9
#define _g _p[10]
#define _g_columnindex 10
#define _ion_ki	*_ppvar[0]._pval
#define _ion_ko	*_ppvar[1]._pval
#define _ion_ik	*_ppvar[2]._pval
#define _ion_dikdv	*_ppvar[3]._pval
 
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
 static void _hoc_alpl(void);
 static void _hoc_alpn(void);
 static void _hoc_betl(void);
 static void _hoc_betn(void);
 static void _hoc_rates(void);
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
 "setdata_kadcr", _hoc_setdata,
 "alpl_kadcr", _hoc_alpl,
 "alpn_kadcr", _hoc_alpn,
 "betl_kadcr", _hoc_betl,
 "betn_kadcr", _hoc_betn,
 "rates_kadcr", _hoc_rates,
 0, 0
};
#define alpl alpl_kadcr
#define alpn alpn_kadcr
#define betl betl_kadcr
#define betn betn_kadcr
 extern double alpl( double );
 extern double alpn( double );
 extern double betl( double );
 extern double betn( double );
 /* declare global and static user variables */
#define a0n a0n_kadcr
 double a0n = 0.1;
#define gml gml_kadcr
 double gml = 1;
#define gmn gmn_kadcr
 double gmn = 0.39;
#define lmin lmin_kadcr
 double lmin = 2;
#define linf linf_kadcr
 double linf = 0;
#define nmin nmin_kadcr
 double nmin = 0.1;
#define ninf ninf_kadcr
 double ninf = 0;
#define pw pw_kadcr
 double pw = -1;
#define q10 q10_kadcr
 double q10 = 5;
#define qq qq_kadcr
 double qq = 5;
#define tq tq_kadcr
 double tq = -40;
#define taun taun_kadcr
 double taun = 0;
#define taul taul_kadcr
 double taul = 0;
#define vhalfl vhalfl_kadcr
 double vhalfl = -56;
#define vhalfn vhalfn_kadcr
 double vhalfn = -1;
#define zetal zetal_kadcr
 double zetal = 3;
#define zetan zetan_kadcr
 double zetan = -1.8;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "vhalfn_kadcr", "mV",
 "vhalfl_kadcr", "mV",
 "a0n_kadcr", "/ms",
 "zetan_kadcr", "1",
 "zetal_kadcr", "1",
 "gmn_kadcr", "1",
 "gml_kadcr", "1",
 "lmin_kadcr", "ms",
 "nmin_kadcr", "ms",
 "pw_kadcr", "1",
 "tq_kadcr", "mV",
 "qq_kadcr", "mV",
 "taul_kadcr", "ms",
 "taun_kadcr", "ms",
 "gkabar_kadcr", "mho/cm2",
 "ik_kadcr", "mA/cm2",
 "gka_kadcr", "mho/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double l0 = 0;
 static double n0 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "vhalfn_kadcr", &vhalfn_kadcr,
 "vhalfl_kadcr", &vhalfl_kadcr,
 "a0n_kadcr", &a0n_kadcr,
 "zetan_kadcr", &zetan_kadcr,
 "zetal_kadcr", &zetal_kadcr,
 "gmn_kadcr", &gmn_kadcr,
 "gml_kadcr", &gml_kadcr,
 "lmin_kadcr", &lmin_kadcr,
 "nmin_kadcr", &nmin_kadcr,
 "pw_kadcr", &pw_kadcr,
 "tq_kadcr", &tq_kadcr,
 "qq_kadcr", &qq_kadcr,
 "q10_kadcr", &q10_kadcr,
 "ninf_kadcr", &ninf_kadcr,
 "linf_kadcr", &linf_kadcr,
 "taul_kadcr", &taul_kadcr,
 "taun_kadcr", &taun_kadcr,
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
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[4]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"kadcr",
 "gkabar_kadcr",
 0,
 "ik_kadcr",
 "gka_kadcr",
 0,
 "n_kadcr",
 "l_kadcr",
 0,
 0};
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 11, _prop);
 	/*initialize range parameters*/
 	gkabar = 0;
 	_prop->param = _p;
 	_prop->param_size = 11;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 5, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[0]._pval = &prop_ion->param[1]; /* ki */
 	_ppvar[1]._pval = &prop_ion->param[2]; /* ko */
 	_ppvar[2]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[3]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _kadistcr_reg() {
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
  hoc_register_prop_size(_mechtype, 11, 5);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 kadcr /mnt/c/Users/analk/OneDrive/Desktop/2021-12-20-ModellingPaper/Fig2/Turi_et_al_2018_o/mechanisms/kadistcr.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double _zqt ;
static int _reset;
static char *modelname = "K-A channel from Klee Ficker and Heinemann";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 () {_reset=0;
 {
   rates ( _threadargscomma_ v ) ;
   Dn = ( ninf - n ) / taun ;
   Dl = ( linf - l ) / taul ;
   }
 return _reset;
}
 static int _ode_matsol1 () {
 rates ( _threadargscomma_ v ) ;
 Dn = Dn  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taun )) ;
 Dl = Dl  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taul )) ;
  return 0;
}
 /*END CVODE*/
 static int states () {_reset=0;
 {
   rates ( _threadargscomma_ v ) ;
    n = n + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taun)))*(- ( ( ( ninf ) ) / taun ) / ( ( ( ( - 1.0 ) ) ) / taun ) - n) ;
    l = l + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taul)))*(- ( ( ( linf ) ) / taul ) / ( ( ( ( - 1.0 ) ) ) / taul ) - l) ;
   }
  return 0;
}
 
static int  rates (  double _lv ) {
   double _la ;
 _la = alpn ( _threadargscomma_ _lv ) ;
   ninf = 1.0 / ( 1.0 + _la ) ;
   taun = betn ( _threadargscomma_ _lv ) / ( _zqt * a0n * ( 1.0 + _la ) ) ;
   if ( taun < nmin ) {
     taun = nmin ;
     }
   _la = alpl ( _threadargscomma_ _lv ) ;
   linf = 1.0 / ( 1.0 + _la ) ;
   taul = 0.26 * ( _lv + 50.0 ) ;
   if ( taul < lmin ) {
     taul = lmin ;
     }
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   _r = 1.;
 rates (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double alpn (  double _lv ) {
   double _lalpn;
 double _lzeta ;
 _lzeta = zetan + pw / ( 1.0 + exp ( ( _lv - tq ) / qq ) ) ;
    _lalpn = exp ( 1.e-3 * _lzeta * ( _lv - vhalfn ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ) ;
    
return _lalpn;
 }
 
static void _hoc_alpn(void) {
  double _r;
   _r =  alpn (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double betn (  double _lv ) {
   double _lbetn;
 double _lzeta ;
 _lzeta = zetan + pw / ( 1.0 + exp ( ( _lv - tq ) / qq ) ) ;
    _lbetn = exp ( 1.e-3 * _lzeta * gmn * ( _lv - vhalfn ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ) ;
    
return _lbetn;
 }
 
static void _hoc_betn(void) {
  double _r;
   _r =  betn (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double alpl (  double _lv ) {
   double _lalpl;
  _lalpl = exp ( 1.e-3 * zetal * ( _lv - vhalfl ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ) ;
    
return _lalpl;
 }
 
static void _hoc_alpl(void) {
  double _r;
   _r =  alpl (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double betl (  double _lv ) {
   double _lbetl;
  _lbetl = exp ( 1.e-3 * zetal * gml * ( _lv - vhalfl ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ) ;
    
return _lbetl;
 }
 
static void _hoc_betl(void) {
  double _r;
   _r =  betl (  *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ki = _ion_ki;
  ko = _ion_ko;
     _ode_spec1 ();
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 ();
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ki = _ion_ki;
  ko = _ion_ko;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_k_sym, _ppvar, 0, 1);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 2);
   nrn_update_ion_pointer(_k_sym, _ppvar, 2, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 3, 4);
 }

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  l = l0;
  n = n0;
 {
   _zqt = pow( q10 , ( ( celsius - 24.0 ) / 10.0 ) ) ;
   rates ( _threadargscomma_ v ) ;
   n = ninf ;
   l = linf ;
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
  ki = _ion_ki;
  ko = _ion_ko;
 initmodel();
 }}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   ek = 25.0 * log ( ko / ki ) ;
   ik = gkabar * n * l * ( v - ek ) ;
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
  ki = _ion_ki;
  ko = _ion_ko;
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
  ki = _ion_ki;
  ko = _ion_ko;
 { error =  states();
 if(error){fprintf(stderr,"at line 76 in file kadistcr.mod:\n	SOLVE states METHOD cnexp\n"); nrn_complain(_p); abort_run(error);}
 } }}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = n_columnindex;  _dlist1[0] = Dn_columnindex;
 _slist1[1] = l_columnindex;  _dlist1[1] = Dl_columnindex;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/mnt/c/Users/analk/OneDrive/Desktop/2021-12-20-ModellingPaper/Fig2/Turi_et_al_2018_o/mechanisms/kadistcr.mod";
static const char* nmodl_file_text = 
  "TITLE K-A channel from Klee Ficker and Heinemann\n"
  ": modified by Brannon and Yiota Poirazi (poirazi@LNC.usc.edu) \n"
  ": to account for Hoffman et al 1997 distal region kinetics\n"
  ": used only in locations > 100 microns from the soma\n"
  ":\n"
  ": modified to work with CVode by Carl Gold, 8/10/03\n"
  ":  Updated by Maria Markaki  12/02/03\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX kadcr\n"
  "	USEION k READ ki, ko WRITE ik 		:Changed from READ ek, 23/04/2010,Nassi\n"
  "  RANGE gkabar,gka,ik\n"
  "  GLOBAL ninf,linf,taul,taun,lmin\n"
  "}\n"
  "\n"
  "\n"
  "UNITS {\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "}\n"
  "\n"
  "\n"
  "PARAMETER {    :parameters that can be entered when function is called in cell-setup   \n"
  "\n"
  "  gkabar = 0      (mho/cm2)  :initialized conductance\n"
  "  vhalfn = -1     (mV)       :activation half-potential (-1), change for pfc, activation at -40\n"
  "  vhalfl = -56    (mV)       :inactivation half-potential\n"
  "  a0n = 0.1       (/ms)      :parameters used\n"
  "  : a0l = 0.05       (/ms)      :parameters used\n"
  "  zetan = -1.8    (1)        :in calculation of\n"
  "  zetal = 3       (1) \n"
  "  :zetal = 3       (1)        :steady state values\n"
  "  gmn   = 0.39    (1)        :and time constants\n"
  "  :gmn   = 0.39    (1)        :and time constants, original\n"
  "  gml   = 1       (1)\n"
  "  lmin  = 2       (ms)\n"
  "  nmin  = 0.1     (ms)\n"
  "  :	nmin  = 0.2     (ms)	:suggested\n"
  "  pw    = -1      (1)\n"
  "  tq    = -40     (mV)\n"
  "  qq    = 5       (mV)\n"
  "  q10   = 5                :temperature sensitivity\n"
  "}\n"
  "\n"
  "\n"
  "ASSIGNED {    :parameters needed to solve DE\n"
  "  v         (mV)\n"
  "  ek        (mV)\n"
  "  celsius  	(degC)\n"
  "  ik        (mA/cm2)\n"
  "  ninf\n"
  "  linf      \n"
  "  taul      (ms)\n"
  "  taun      (ms)\n"
  "  gka       (mho/cm2)\n"
  "  ki		    (mM)\n"
  "  ko		    (mM)\n"
  "}\n"
  "\n"
  "\n"
  "STATE {       :the unknown parameters to be solved in the DEs \n"
  "	n l\n"
  "}\n"
  "\n"
  ": Solve qt once in initial block\n"
  "LOCAL qt\n"
  "\n"
  "INITIAL {    :initialize the following parameter using rates()\n"
  "  qt = q10^((celsius-24)/10(degC))       : temperature adjustment factor\n"
  "	rates(v)\n"
  "	n=ninf\n"
  "	l=linf\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE states METHOD cnexp\n"
  "	ek = 25 * log(ko/ki)		:Changed, added, 23/04/2010, Nassi\n"
  "	ik = gkabar*n*l*(v-ek)\n"
  "}\n"
  "\n"
  "\n"
  "DERIVATIVE states {     : exact when v held constant; integrates over dt step\n"
  "  rates(v)              : do this here\n"
  "  n' = (ninf - n)/taun\n"
  "  l' = (linf - l)/taul\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "PROCEDURE rates(v (mV)) {		 :callable from hoc\n"
  "	LOCAL a\n"
  "\n"
  "  a = alpn(v)\n"
  "  ninf = 1/(1 + a)		             : activation variable steady state value\n"
  "  taun = betn(v)/(qt*a0n*(1+a))	   : activation variable time constant\n"
  "	if (taun<nmin) {taun=nmin}	     : time constant not allowed to be less than nmin\n"
  "    a = alpl(v)\n"
  "    linf = 1/(1 + a)               : inactivation variable steady state value\n"
  "  	:taul = 6 (ms)\n"
  "  	taul = 0.26(ms/mV)*(v+50)      : inactivation variable time constant (0.26)\n"
  "	if (taul<lmin) {taul=lmin}       : time constant not allowed to be less than lmin\n"
  "}\n"
  "\n"
  "\n"
  "FUNCTION alpn(v(mV)) { LOCAL zeta\n"
  "  zeta = zetan+pw/(1+exp((v-tq)/qq))\n"
  "UNITSOFF\n"
  "  alpn = exp(1.e-3*zeta*(v-vhalfn)*9.648e4/(8.315*(273.16+celsius))) \n"
  "UNITSON\n"
  "}\n"
  "\n"
  "FUNCTION betn(v(mV)) { LOCAL zeta\n"
  "  zeta = zetan+pw/(1+exp((v-tq)/qq))\n"
  "UNITSOFF\n"
  "  betn = exp(1.e-3*zeta*gmn*(v-vhalfn)*9.648e4/(8.315*(273.16+celsius))) \n"
  "UNITSON\n"
  "}\n"
  "\n"
  "FUNCTION alpl(v(mV)) {\n"
  "UNITSOFF\n"
  "  alpl = exp(1.e-3*zetal*(v-vhalfl)*9.648e4/(8.315*(273.16+celsius))) \n"
  "UNITSON\n"
  "}\n"
  "\n"
  "FUNCTION betl(v(mV)) {\n"
  "UNITSOFF\n"
  "  betl = exp(1.e-3*zetal*gml*(v-vhalfl)*9.648e4/(8.315*(273.16+celsius))) \n"
  "UNITSON\n"
  "}\n"
  "\n"
  ;
#endif
