/*************************************************************************
    > File Name: admm.cpp
    > Author: ling fang
    > Mail: fangl@bayescom.com 
 ************************************************************************/

#include "admm.h"

namespace alg
{
using namespace std;

ADMM::ADMM(struct args_t a_)
{
	a = a_;
	
	numData = a.numData;
	numFeatures = a.numFeatures;
	numIns = a.insLen;
	
	numProcs = a.numProcs;
	maxIter = a.admm_maxIter;
	rho = a.rho;
	l1reg = a.l1reg;
	l2reg = a.l2reg;
	
	x     = new double[numFeatures];
	y     = new double[numFeatures];
	z     = new double[numFeatures];
	z_pre = new double[numFeatures];
	w     = new double[numFeatures];
	g     = new double[numFeatures];
	
	for(uint32_t i = 0;i < numFeatures;i++)
	{
		g[i] = 0.0;
	}
}

ADMM::~ADMM()
{
	if(x != NULL)
	{
		delete [] x;
		x = NULL;
	}
	
	if(z != NULL)
	{
		delete [] z;
		z = NULL;
	}
	
	if(z_pre != NULL)
	{
		delete [] z_pre;
		z_pre = NULL;
	}
	
	if(y != NULL)
	{
		delete [] y;
		y = NULL;
	}
	
	if(w != NULL)
	{
		delete [] w;
		w = NULL;
	}
}

void ADMM::train()
{
	int k = 0;
	double progressiveLoss = 0;
	double loss;
	
	string data_file;
	stringstream stream;
	stream << a.myid;

	string myid = stream.str();
	data_file = a.trainFile + "_" + myid;
//	cout << "[" << a.myid << "]" << " reading " << data_file << endl;
	ifstream train_stream(data_file);
	
	if(train_stream.fail()){
		cout << "read train file error!" << endl;
		return;
	}
	
	if(a.isRoot)
		printf("%3s %10s %10s %10s %10s %10s\n", "#", "r norm", "eps_pri", "s norm", "eps_dual","logloss");
	while(k < maxIter){
		if(a.isRoot)
			printf("%3d ",k);
		if(!x_update(train_stream,&loss))
			break;
		z_update();
		y_update();
		/* Termination checks */
		if(isStop())
			break;
		progressiveLoss += loss;
		printf("%10.4f\n",progressiveLoss);
		k++;
	}

	train_stream.close();
	saveModel();
}

bool ADMM::isStop()
{
	double send[3] = {0};
	double recv[3] = {0};

	for(uint32_t i = 0;i < numFeatures;i++){
		send[0] += (x[i] - z[i]) * (x[i] - z[i]);
		send[1] += (x[i]) * (x[i]);
		send[2] += (y[i]) * (y[i]);
	}

	MPI_Allreduce(send,recv,3,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

	double prires  = sqrt(recv[0]);  /* sqrt(sum ||r_i||_2^2) */
	double nxstack = sqrt(recv[1]);  /* sqrt(sum ||x_i||_2^2) */
	double nystack = sqrt(recv[2]);  /* sqrt(sum ||y_i||_2^2) */

	double zdiff = 0.0;
	double z_squrednorm = 0.0; 
	for(uint32_t i = 0;i < numFeatures;i++){
		zdiff += (z[i] - z_pre[i]) * (z[i] - z_pre[i]);
		z_squrednorm += z[i] * z[i];
	}

	double z_norm = sqrt(numProcs) * sqrt(z_squrednorm);
 	double dualres = sqrt(numProcs) * rho * sqrt(zdiff); /* ||s^k||_2^2 = N rho^2 ||z - zprev||_2^2 */
	//double vmax = nxstack > z_norm?nxstack:z_norm;

	double eps_pri  = sqrt(numProcs*numData)*ABSTOL + RELTOL * fmax(nxstack,z_norm);
	double eps_dual = sqrt(numProcs*numData)*ABSTOL + RELTOL * nystack;

	if (a.isRoot) {
		printf("%10.4f %10.4f %10.4f %10.4f ", prires, eps_pri, dualres, eps_dual);
	}

//	if (prires <= eps_pri && dualres <= eps_dual) {
//		return true;
//	}
	return false;
}

double ADMM::logloss(double p,double y){
	p = max(min(p, 1. - 1e-15), 1e-15);
	if(y-1 < EPSILON && y-1>EPSILON*(-1)){
		return (-1)*log(p);    
	}else{
		return (-1)*log(1.-p);
	}
}

double ADMM::innerProduct(vector<uint32_t> train_ins)
{
	double innerP = 0.0;
	for(size_t i = 0;i < train_ins.size();i++)
	{
		innerP += x[train_ins[i]];
	}
	return innerP;
}

double ADMM::sigmoid(double inx)
{
	return 1.0 / (1.0 + std::exp(-inx));
}

void ADMM::get_Grad(vector<uint32_t> train_ins,int label)
{
	double innerP = innerProduct(train_ins);
	double predict = sigmoid(innerP);
	
	double grad = predict - label;
	
	for(uint32_t index = 0;index < numIns;index++){
		g[train_ins[index]] += grad;
	}
	
	//the reduial grad
	for(uint32_t i = 0;i < numFeatures;i++){
		g[i] += rho * (x[i] - z[i]) + y[i];
	}
}

double ADMM::predict(vector<uint32_t> train_ins)
{
	double inner = innerProduct(train_ins);
	return sigmoid(inner);
}

bool ADMM::x_update(ifstream &train_stream,double *loss){
	//sgd train
	double beta = 0.01;
	string line;
	vector<uint32_t> train_ins;
	int label;
	vector<string> fields;
	
	if(getline(train_stream,line))
	{
		util::str_util::split(line, " ", fields);
		label = atoi(fields[0].c_str());
		for(uint32_t j = 0;j < numIns;j++){
			train_ins.push_back((uint32_t)atoi(fields[j+1].c_str()));
		}
	}else{
		return false;
	}
	
	get_Grad(train_ins,label);
	for(uint32_t i = 0;i < numFeatures;i++)
	{
		x[i] = x[i] - beta * g[i];
	}
	
	*loss = logloss(predict(train_ins),label);
	
	return true;
}

void ADMM::y_update(){
	for(uint32_t i = 0;i < numFeatures;i++){
		y[i] += rho * (x[i] - z[i]);
	}
}

void ADMM::z_update(){
	double s = 1.0/(rho * numProcs + 2*l2reg);
	double t = s * l1reg;

	for(uint32_t i = 0;i < numFeatures;i++)
	{
		w[i] = s * (x[i] + y[i]);
		z_pre[i] = z[i];
	}
	
	MPI_Allreduce(w, z,  numFeatures, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	
	if(hasL1reg)
		softThreshold(t,z);
}

void ADMM::softThreshold(double t_,double *z)
{
	for(uint32_t i = 0;i < numFeatures;i++){
		if(z[i] > t_){
			z[i] -= t_;
		}else if(z[i] <= t_ && z[i] >= -t_){
			z[i] = 0.0;
		}else{
			z[i] += t_;
		}
	}
}

void ADMM::saveModel(){
	ofstream of("model.dat");
	for(uint32_t i = 0;i < numFeatures;i++){
		of << z[i] << endl;
	}	
	of.close();
}

}
