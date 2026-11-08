#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include "linear.h"

//#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define Malloc(type,n) (type *)R_Calloc(n,type)

void print_null(const char *s) {}

void setup_params(int *type, double *cost, double *epsilon, double* svr_eps, int *nrWi, double *Wi, int *WiLabels, int *cross, int *verbose, int *findC, int *useInitC, int *regularizeBias, double *nu);
void setup_problem(double *X, double *Y, int *nbSamples, int *nbDim, int *sparse, int *rowindex, int *colindex, double *bi, int *verbose);
double do_cross_validation(void);
double do_find_parameter_C(void);

struct feature_node *x_space;
struct parameter param;
struct problem prob;
struct model* model_;
int flag_cross_validation;
int nr_fold;
int flag_find_C;
int flag_C_specified;
int flag_verbose;
// Phase B: set by do_find_parameter_C() when it searches svr_eps too
// (L2R_L2LOSS_SVR / type 11 only); read back by trainLinear() into its
// own dedicated, safely-sized output parameter (bestP_ret) rather than
// into W_ret, whose buffer can be as small as a single element (e.g.
// bias<=0 and a single feature) and so cannot safely carry a second
// value.
double last_best_p;

/**
 * Function: trainLinear
 *
 * Author: Thibault Helleputte
 *
 */
void trainLinear(double *W_ret, int* labels_ret, double *X, double *Y, int *nbSamples, int *nbDim, int *sparse, int *rowindex, int *colindex,
                 double *bi, int *type, double *cost, double *epsilon, double* svr_eps, int *nrWi, double *Wi, int *WiLabels, int *cross, int *verbose, int *findC, int *useInitC, int *regularizeBias, double *bestP_ret, double *nu)
{
	if(*verbose)
	    flag_verbose = 1;
	else
	    flag_verbose = 0;

	const char *error_msg;

	setup_params(type, cost, epsilon, svr_eps, nrWi, Wi, WiLabels, cross, verbose, findC, useInitC, regularizeBias, nu);
	setup_problem(X, Y, nbSamples, nbDim, sparse, rowindex, colindex, bi, verbose);

	if(*verbose)
		Rprintf("SETUP CHECK\n");
	
	error_msg = check_parameter(&prob,&param);

	if(error_msg){
		Rprintf("ERROR: %s\n",error_msg);
		// setup_problem() above already allocated prob.y/prob.x/x_space; free
		// them on this path too, since it returns before the normal cleanup below.
		R_Free(prob.y);
		R_Free(prob.x);
		R_Free(x_space);
		return;
	}
	
	if (flag_find_C)
	{
		if(*verbose)
			Rprintf("FIND C\n");

		last_best_p = NA_REAL;
		W_ret[0] = do_find_parameter_C();
		*bestP_ret = last_best_p;
	}
	else if(flag_cross_validation)
	{
		if(*verbose)
			Rprintf("CROSS VAL\n");
		
		W_ret[0] = do_cross_validation();
	}
	else
	{
		if(*verbose)
			Rprintf("TRAIN\n");
		
		model_=train(&prob, &param);
		copy_model(W_ret, labels_ret, model_);
		free_and_destroy_model(&model_);
	}
	if(*verbose)
		Rprintf("FREE SPACE\n");
	
	//No need to destroy param because its members are shallow copies of Wi and WiLabels
	//destroy_param(&param);
	R_Free(prob.y);
	R_Free(prob.x);
	R_Free(x_space);
	
	if(*verbose)
		Rprintf("FREED SPACE\n");
	

	return;
}

double do_find_parameter_C(void)
{
	// find_parameter_C() (single-purpose: C-only search) was renamed and
	// restructured upstream (LIBLINEAR v2.30) into find_parameters(),
	// which also supports a joint C/svr_eps search for L2R_L2LOSS_SVR.
	// Phase B exposes that: findC is now also validated above to reach
	// here for L2R_L2LOSS_SVR (type 11), in addition to the original
	// L2R_LR/L2R_L2LOSS_SVC (types 0/2). For type 11, start_p reuses the
	// same 'useInitC' flag/semantics as start_C already does for C --
	// "use my given value (svr_eps, in param.p) as the search anchor" --
	// rather than introducing a second, near-duplicate R-level argument
	// just for p. For types 0/2, find_parameters()'s own internal
	// dispatch never touches its p-search branch at all regardless of
	// start_p, and always sets best_p to the sentinel -1. best_rate was
	// renamed best_score, same meaning (fraction correct for
	// classification, negative MSE for regression -- see below) for
	// these solver types; max_C is no longer a caller-supplied argument
	// -- find_parameters() now applies the same 1024 default internally.
	double start_C, start_p, best_C, best_p, best_score;
	if (flag_C_specified)
		start_C = param.C;
	else
		start_C = -1.0;

	if (param.solver_type == L2R_L2LOSS_SVR && flag_C_specified)
		start_p = param.p;
	else
		start_p = -1.0;

	if(flag_verbose)
		Rprintf("Doing parameter search with %d-fold cross validation.\n", nr_fold);

	find_parameters(&prob, &param, nr_fold, start_C, start_p, &best_C, &best_p, &best_score);

	if(flag_verbose)
		Rprintf("Best C = %g  Best p = %g  CV score = %g\n", best_C, best_p, best_score);

	if(best_score == 0){
		last_best_p = NA_REAL;
		return NA_REAL;
	}

	if(param.solver_type == L2R_L2LOSS_SVR)
		last_best_p = best_p;

	return best_C;
}

/**
 * Function: do_cross_validation
 *
 * Author: Thibault Helleputte
 *
 */
double do_cross_validation(void)
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double *target = Malloc(double, prob.l);
	double res;
	
	cross_validation(&prob,&param,nr_fold,target);
	if(param.solver_type == L2R_L2LOSS_SVR ||
	   param.solver_type == L2R_L1LOSS_SVR_DUAL ||
	   param.solver_type == L2R_L2LOSS_SVR_DUAL)
	{
		double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		res=total_error/prob.l;
		//squared_correlation_coefficient = 
		//		((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
		//		((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		res = 1.0*total_correct/prob.l;
	}
	
	R_Free(target);
	return(res);
}

/**
 * Function: setup_params
 *     Replaces parse_command_line from train.c
 * Author: Pierre Gramme, Jerome Paul
 *
 */
void setup_params(int *type, double *cost, double *epsilon, double* svr_eps, int *nrWi, double *Wi, int *WiLabels, int *cross, int *verbose, int *findC, int *useInitC, int *regularizeBias, double *nu)
{
	if(*verbose){
		Rprintf("ARGUMENTS SETUP\n");
	}

	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// ARGUMENTS SETUP
	param.solver_type = *type;
	param.C = *cost;
	param.p = *svr_eps;
	param.eps = *epsilon;
	//Note: bias will be set in setup_problem(...)
	param.nr_weight = *nrWi;
	//TODO: deep copy might be safer than pointer copy
	param.weight_label = WiLabels;
	param.weight = Wi;

	// Phase B: both regularize_bias and nu (the -R and -n CLI flags) are
	// now exposed at the R level, as 'regularizeBias' (default TRUE,
	// restoring the Stage (ii) behavior for anyone who doesn't touch it)
	// and 'nu' (default 0.5, matching upstream's own train.c CLI
	// default; only meaningful for type=21, ONECLASS_SVM, which is now
	// R-reachable too -- see setup_problem()/check_parameter() for the
	// rest of one-class SVM's own validation). R-level LiblineaR()
	// already enforces check_parameter()'s own constraints (bias==1 for
	// regularizeBias=FALSE, allowed solver types) proactively with a
	// clear stop() before this is ever reached, but check_parameter()
	// below is still the final authority.
	param.regularize_bias = *regularizeBias;
	param.nu = *nu;

	if(*cross>0)
	{
		flag_cross_validation = 1; 
		nr_fold = *cross;
	}
	else
	{
		flag_cross_validation = 0; 
		nr_fold = 0;
	}

	if(*useInitC){ // only for do_find_parameter_C() (calls upstream's find_parameters())
		flag_C_specified = 1;
	} else {
		flag_C_specified = 0;
	}
	
	if(*findC){
		flag_find_C = 1;
		
		if(!flag_cross_validation)
			nr_fold = 5;
		
		if(param.solver_type != L2R_LR && param.solver_type != L2R_L2LOSS_SVC && param.solver_type != L2R_L2LOSS_SVR){
			error("Warm-start parameter search only available for L2R_LR, L2R_L2LOSS_SVC and L2R_L2LOSS_SVR\n");
		}
	} else {
		flag_find_C = 0;
	}

	// Verbose or not?
	if(!*verbose){
		// liblinear_print_string = &print_null;
		print_func = &print_null;
	}
	
	set_print_string_function(print_func);
	
	// NA value for eps is coded as <=0 instead of INF in original code
	//TODO in 1.94: update
	if(param.eps <= 0)
	{
		switch(param.solver_type)
		{
			case L2R_LR:
			case L2R_L2LOSS_SVC:
				param.eps = 0.01;
				break;
			case L2R_L2LOSS_SVR:
				param.eps = 0.001;
				break;
			case L2R_L2LOSS_SVC_DUAL:
			case L2R_L1LOSS_SVC_DUAL:
			case MCSVM_CS:
			case L2R_LR_DUAL:
				param.eps = 0.1;
				break;
			case L1R_L2LOSS_SVC:
			case L1R_LR:
				param.eps = 0.01;
				break;
			case L2R_L1LOSS_SVR_DUAL:
			case L2R_L2LOSS_SVR_DUAL:
				param.eps = 0.1;
				break;
			// Phase B: type=21 (ONECLASS_SVM) is now R-reachable. Without
			// this case, epsilon=NULL would leave param.eps at its -1
			// sentinel (invalid, would fail check_parameter's eps<=0
			// rejection) instead of routing to a real default, unlike
			// every other solver type. Matches upstream's own train.c
			// CLI default for -s 21 (0.01, called out as the one
			// exception in its help text: "Dual maximal violation <= eps
			// ... default 0.1 except 0.01 for -s 21").
			case ONECLASS_SVM:
				param.eps = 0.01;
				break;
		}
	}
}


/**
 * Function: setup_problem
 *     Replaces read_problem from train.c
 * Author: Pierre Gramme
 *
 */
void setup_problem(double *X, double *Y, int *nbSamples, int *nbDim, int *sparse, int *rowindex, int *colindex, 
                 double *bi, int *verbose)
{
	int i, j, k, max_index;
	i=j=k=0;
	
	if(*verbose){
		Rprintf("PROBLEM SETUP\n");
	}
	
	// PROBLEM SETUP
	prob.l = *nbSamples;
	prob.bias = *bi;
	
	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	
	// Compute the x_space allocation size in a 64-bit-safe type: for large
	// dense n*p (e.g. wide-feature genomics/text data), the equivalent 32-bit
	// int arithmetic can overflow and under-allocate, so the size is checked
	// against INT_MAX below before it is narrowed for the actual allocation.
	double allocSizeD = (double)(*nbDim) * prob.l + prob.l;
	if (*sparse > 0){
		allocSizeD = (double)rowindex[prob.l] + prob.l;
	}
	if(prob.bias >= 0)
		allocSizeD += prob.l;
	if(allocSizeD > (double) INT_MAX)
		error("Dataset too large for LiblineaR: requested allocation (%.0f feature-node entries) exceeds the supported range.", allocSizeD);

	int allocSize = (int) allocSizeD;
	if (*sparse > 0 && *verbose)
		Rprintf("allocSize: %d\n",allocSize);

	 x_space = Malloc(struct feature_node,allocSize);
	
	
	if(*verbose){
		Rprintf("FILL DATA STRUCTURE\n");
	}
	// Fill data stucture
	max_index = 0;
	k=0;
    if(*sparse > 0){
        // handle sparse matrix
        int totalK = 0;
        for(i=0; i<prob.l; i++){
            prob.y[i] = Y[i];
            prob.x[i] = &x_space[k];

            int nnz = rowindex[i+1]-rowindex[i];
            for(j=0; j<nnz; j++, k++, totalK++){
                x_space[k].index = colindex[totalK];
                x_space[k].value = X[totalK];

                if(colindex[totalK] > max_index){
                    max_index = colindex[totalK];
                }
            }

            if(prob.bias >= 0)
                x_space[k++].value = prob.bias;
            x_space[k++].index = -1;
        }
    }
    else {
        for(i=0;i<prob.l;i++){
            prob.y[i] = Y[i];
            prob.x[i] = &x_space[k];

            for(j=1;j<*nbDim+1;j++){
                if(X[(*nbDim*i)+(j-1)]!=0){
                    x_space[k].index = j;
                    x_space[k].value = X[(*nbDim*i)+(j-1)];
                    k++;
                    if(j>max_index){
                        max_index=j;
                    }
                }
            }
            if(prob.bias >= 0)
                x_space[k++].value = prob.bias;
            x_space[k++].index = -1;
        }
    }

	if(prob.bias >= 0)
	{
		prob.n=max_index+1;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n;
		x_space[k-2].index = prob.n;
	}
	else
		prob.n=max_index;
}
