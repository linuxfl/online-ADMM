#ifndef ARGS_H
#define ARGS_H

#include <iostream>
#include <string>

/*
	struct args_t include param as following:
		process: myid,numProcs,isRoot
		Logistic Regression: 
		LBFGS:
		ADMM:
		Traning Set:
*/
struct args_t
{
	//process param
	int myid;
	int numProcs;
	bool isRoot;
	
	//LBFGS
	int lbfgs_maxIter;
	//debug
	bool lbfgsDebug;
	double epsilon;
	int m;
	
	//Training set
	std::string trainFile;
//	std::string testFile;
	
	uint32_t numData;
	uint32_t numFeatures;
	uint32_t insLen;
	
	//ADMM
	double rho;
	int admm_maxIter;
	bool hasL1reg;
	
	//Logistic Regression
	double l1reg;
	double l2reg;
};
void initialize_args(args_t& a);
bool get_args(struct args_t &a,const char *file,int myid,int numprocs);
void print_args(struct args_t args);

#endif
