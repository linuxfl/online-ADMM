/*************************************************************************
    > File Name: args.cpp
    > Author: ling fang
    > Mail: fangl@bayescom.com 
 ************************************************************************/
 
#include <iostream>
#include <cstdio>
#include <string>
#include <unistd.h>

#include "conf_util.h"
#include "args.h"

using namespace std;

void initialize_args(struct args_t& a) {
	// processors
	a.myid = -1;
	a.numProcs = -1;
	a.isRoot = false;
    
	a.lbfgs_maxIter = 0;
	a.epsilon = 0;
	
	// data sets
//	a.trainFile = NULL;  
//    	a.testFile = NULL; 
    
	// data set attributes
	a.numFeatures = -1;
	a.numData = -1;
	a.insLen = -1;
	
	a.l1reg = 0.0;
	a.l2reg = 0.0;
}

bool get_args(struct args_t &a,const char *file,int myid,int numprocs)
{

	//test the conf file exist
	if(access(file,R_OK) == -1){
		cout << "The conf file is ureadable..." << endl;
		return false;
	}
	string conf_file(file);
	util::conf_util admm_conf;
	admm_conf.parse(conf_file);
	
	string train_file = admm_conf.getItem<string>("train_file");
	//string test_file = admm_conf.getItem<string>("model_file");
	
	double epsilon = admm_conf.getItem<double>(string("epsilon"));
	int lbfgs_maxIter = admm_conf.getItem<int>(string("lbfgs_maxIter"));
	bool lbfgsDebug = admm_conf.getItem<bool>(string("lbfgs_debug"));

	double l1reg = admm_conf.getItem<double>(string("l1reg"));
	double l2reg = admm_conf.getItem<double>(string("l2reg"));
	bool hasL1reg = admm_conf.getItem<bool>(string("hasL1reg"));	

	int numFeatures = admm_conf.getItem<int>(string("num_fea"));
	int insLen = admm_conf.getItem<int>(string("ins_len"));
	int m = admm_conf.getItem<int>(string("m"));

	double rho = admm_conf.getItem<double>(string("rho"));
	int admm_maxIter = admm_conf.getItem<int>(string("admm_maxIter"));	
	
	a.myid = myid;
	a.numProcs = numprocs;
	a.isRoot = (myid == 0);
    
	a.lbfgs_maxIter = lbfgs_maxIter;
	a.epsilon = epsilon;
	a.m = m;
	
    	// data sets
	a.trainFile = train_file;  
    
    	// data set attributes
    	a.numFeatures = (uint32_t)numFeatures;
	a.insLen = (uint32_t)insLen;
   	//a.numData = -1;

	a.lbfgsDebug = lbfgsDebug;
	//admm alg
	a.hasL1reg = hasL1reg;	
	a.l1reg = l1reg;
	a.l2reg = l2reg;
	a.rho = rho;	
	a.admm_maxIter = admm_maxIter;
	
	return true;
}

void print_args(struct args_t args)
{
	cout << "#**************Configuration**************" << endl;
	cout << "#Number of processors: " << args.numProcs << endl;
	
	// data sets
	cout << "#Training file: " << args.trainFile  << endl;
	cout << "#Number of features: " << args.numFeatures << endl;
	cout << "#Length of instance: " << args.insLen << endl;
	
    	//admm
	cout << "#hasL1reg: " << args.hasL1reg << endl;
	cout << "#l2reg: " << args.l2reg << endl;
	cout << "#l1reg: " << args.l1reg << endl;
	cout << "#rho: " << args.rho << endl;
	cout << "#admm_maxIter: " << args.admm_maxIter << endl;
	
	//lbfgs
	cout << "#lbfgs_maxIter: " << args.lbfgs_maxIter << endl;
	cout << "#epsilon: " << args.epsilon << endl;
	cout << "#lbfgs_debug: " << args.lbfgsDebug << endl;
	cout << "#m: " << args.m << endl;
    
	cout << "#*****************************************" << endl;
}
