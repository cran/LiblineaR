#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "linear.h"
//#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define Malloc(type,n) (type *)Calloc(n,type)

void print_null(const char *s) {}
double do_cross_validation(void);

struct feature_node *x_space;
struct parameter param;
struct problem prob;
struct model* model_;
int flag_cross_validation;
int nr_fold;
double bias;
	

void trainLinear(double *W, double *X, double *Y, int *nbSamples, int *nbDim, double *bi, int *type, double *cost, double *epsilon, int *nrWi, double *Wi, int *WiLabels, int *cross, int *verbose);


/**
 * Function: trainLinear
 *
 * Author: Thibault Helleputte
 *
 */
void trainLinear(double *W, double *X, double *Y, int *nbSamples, int *nbDim, double *bi, int *type, double *cost, double *epsilon, int *nrWi, double *Wi, int *WiLabels, int *cross, int *verbose){
	
	void (*print_func)(const char*) = NULL;	// default printing to stdout
	const char *error_msg;
	int i, j, k, max_index;
	i=j=k=0;
	bias = -1;
	
	if(*verbose){
		Rprintf("ARGUMENTS SETUP\n");
	}
	// ARGUMENTS SETUP
	param.solver_type = *type;
	param.C = *cost;
	// Verbose or not?
	if(!*verbose){
//		liblinear_print_string = &print_null;
		print_func = &print_null;
	}
	set_print_string_function(print_func);
	if(*epsilon <= 0){
		if(param.solver_type == L2R_LR || param.solver_type == L2R_L2LOSS_SVC)
			param.eps = 0.01;
		else if(param.solver_type == L2R_L2LOSS_SVC_DUAL || param.solver_type == L2R_L1LOSS_SVC_DUAL || param.solver_type == MCSVM_CS || param.solver_type == L2R_LR_DUAL)
			param.eps = 0.1;
		else if(param.solver_type == L1R_L2LOSS_SVC || param.solver_type == L1R_LR)
			param.eps = 0.01;
	}
	else
		param.eps=*epsilon;
		
	param.nr_weight = *nrWi;
	param.weight_label = WiLabels;
	param.weight = Wi;
	
	if(*cross>0){
		flag_cross_validation = 1; 
		nr_fold = *cross;
	}
	else{
		flag_cross_validation = 0; 
		nr_fold = 0;
	}
	
	if(*verbose){
		Rprintf("PROBLEM SETUP\n");
	}
	// PROBLEM SETUP
	prob.l = *nbSamples;
	bias = *bi;
	prob.bias = *bi;
	
	prob.y = Malloc(int,prob.l);
	prob.x = Malloc(struct feature_node *,prob.l);
	
	if(prob.bias >= 0)
		x_space = Malloc(struct feature_node,(*nbDim+1)*(*nbSamples)+prob.l);
	else
		x_space = Malloc(struct feature_node,(*nbDim)*(*nbSamples)+prob.l);
	
	if(*verbose){
		Rprintf("FILL DATA STRUCTURE\n");
	}
	// Fill data stucture
	max_index = 0;
	k=0;
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
	
	if(prob.bias >= 0){
		prob.n=max_index+1;
		for(i=1;i<prob.l;i++)
			(prob.x[i]-2)->index = prob.n; 
		x_space[k-2].index = prob.n;
	}
	else
		prob.n=max_index;
	
	if(*verbose){
		Rprintf("SETUP CHECK\n");
	}
	// SETUP CHECK
	error_msg = NULL;
	
	error_msg = check_parameter(&prob,&param);
	
	if(error_msg){
		Rprintf("Error: %s\n",error_msg);
		return;
	}
	
	if(flag_cross_validation){
		if(*verbose){
			Rprintf("CROSS VAL\n");
		}
		//do_cross_validation();
		W[0]=do_cross_validation();
	}
	else{
		if(*verbose){
			Rprintf("TRAIN\n");
		}
		model_=train(&prob, &param);
		if(*verbose){
			Rprintf("COPY RESULT FOR ");
		}
		if(model_->nr_class==2){
			if(*verbose){
				Rprintf("TWO CLASSES\n");
			}
			for(i=0; i<*nbDim; i++){
				W[i]=model_->w[i];
			}
			if(prob.bias >= 0){
				W[*nbDim]=model_->w[i];
			}
		}
		else{
			if(*verbose){
				Rprintf("%d CLASSES\n",model_->nr_class);
			}
			for(i=0;i<model_->nr_class;i++){
				if(prob.bias >= 0){
					for(j=0; j<*nbDim+1; j++){
						W[(*nbDim+1)*i+j]=model_->w[(*nbDim+1)*i+j];
					}
				}
				else{
					for(j=0; j<*nbDim; j++){
						W[*nbDim*i+j]=model_->w[*nbDim*i+j];
					}
				}	
			}
		}
		free_and_destroy_model(&model_);
	}
	if(*verbose){
		Rprintf("FREE SPACE\n");
	}
	//destroy_param(&param);
	Free(prob.y);
	Free(prob.x);
	Free(x_space);

	return;
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
	int *target = Malloc(int, prob.l);
	cross_validation(&prob,&param,nr_fold,target);
	for(i=0;i<prob.l;i++)
		if(target[i] == prob.y[i])
			++total_correct;
	Free(target);
	return(1.0*total_correct/prob.l);
}
