/*************************************************************************
    > File Name: train.cpp
    > Author: ling fang
    > Mail: fangl@bayescom.com 
 ************************************************************************/
 
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <string>
#include <ctime>

#include "args.h"
#include "admm.h"

using namespace std;
using namespace alg;

void run(struct args_t a);

int main(int argc,char **argv)
{
	if(argc < 2){
		cout << "Usage:train conf_file" << endl;
		return 0;
	}
	
	// initialize MPI
	int myid, numprocs;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	
	time_t begin_time,end_time;
	time(&begin_time);
	// parse command line args
	struct args_t a;
	initialize_args(a);
	
	// get args from conf file
	if (!get_args(a,argv[1],myid, numprocs)) {
	    // if bad args, print help and exit
        	if (a.isRoot){
			fprintf(stderr,"Enter the conf file %s is not exist!",argv[1]);
		}
        	MPI_Finalize();
        	return EXIT_FAILURE;
    	}
    
    	// if good args, print parameters
   	 if (a.isRoot) print_args(a);
    
    	// run and exit
    	run(a);
	
	time(&end_time);
	
	if(a.isRoot)
		cout << "Total train time is " << difftime(end_time , begin_time) << " second." << endl;
    	MPI_Finalize();
	return EXIT_SUCCESS;
}

void run(struct args_t a)
{
	//ADMM train
	if(a.isRoot)
		cout << "Begin to train..." << endl;
	ADMM admm(a);
	admm.train();
	if(a.isRoot)
		cout << "Finish..." << endl;
}
