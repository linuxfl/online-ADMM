#ifndef ADMM_H
#define ADMM_H

#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <cmath>
#include <vector>

#include "args.h"
#include "str_util.h"

#define EPSILON 1e-6

namespace alg
{
using namespace std;

class ADMM{
	public:
		ADMM(struct args_t a_);
		~ADMM();
		
		bool x_update(ifstream &,double *);
		void y_update();
		void z_update();
		void train();
		void softThreshold(double ,double *);
		void saveModel();
		bool isStop();

	public:
		double innerProduct(vector<uint32_t>);
		double sigmoid(double);
		void get_Grad(vector<uint32_t>,int);
		double logloss(double p,double y);
		double predict(vector<uint32_t> train_ins);
		
	private:
		struct args_t a;
		
		uint32_t numData;
		uint32_t numFeatures;
		uint32_t numIns;
		
		int numProcs;
		int maxIter;

		double l1reg;
		double l2reg;
		double rho;
		bool hasL1reg;
		
		double *x;
		double *z;
		double *z_pre;
		double *y;
		double *w;
		double *g;

		const double RELTOL = 1e-2;
		const double ABSTOL = 1e-4;

		double send[3];
		double recv[3];

		
};

}
#endif
