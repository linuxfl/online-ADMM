/*************************************************************************
    > File Name: admm.cpp
    > Author: ling fang
    > Mail: fangl@bayescom.com 
 ************************************************************************/

#include "admm.h"
#include <ctime>
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

	time_t x_begin_time,x_end_time,y_begin_time,y_end_time,z_begin_time,z_end_time;
	
	time(&y_begin_time);
	time(&z_begin_time);

	int k = 0;
	double progressiveLoss = 0;
	double loss;
	
	string data_file;
	stringstream stream;
	stream << a.myid;

	string myid = stream.str();
	data_file = a.trainFile + "_" + myid;
	cout << "[" << a.myid << "]" << " reading " << data_file << endl;
	ifstream train_stream(data_file);
	
	if(train_stream.fail()){
		cout << "read train file error!" << endl;
		return;
	}
	
	string line;
	vector<uint32_t> train_ins;
	int label;
	vector<string> fields;

	while(k < maxIter){

		if(getline(train_stream,line))
		{
			train_ins.clear();
			fields.clear();
			util::str_util::split(line, " ", fields);
			label = atoi(fields[0].c_str());
			for(uint32_t j = 0;j < numIns;j++){
				train_ins.push_back((uint32_t)atoi(fields[j+1].c_str()));
			}
		}else{
			break;
		}
		time(&x_begin_time);
		if(!x_update(train_ins,label,&loss))
			break;
		time(&x_end_time);
		if(a.isRoot)
			cout << "x train time is :" << difftime(x_end_time , x_begin_time) << " second." << endl;
		time(&z_begin_time);
		z_update(train_ins);
		time(&z_end_time);
		if(a.isRoot)
			cout << "z train time is :" << difftime(z_end_time , z_begin_time) << " second." << endl;
		
		time(&y_begin_time);
		y_update(train_ins);
		time(&y_end_time);
		if(a.isRoot)
			cout << "y train time is :" << difftime(y_end_time , y_begin_time) << " second." << endl;
		progressiveLoss += loss;
		if(a.isRoot)
			printf("ProgressiveLoss : %10.4f\n",progressiveLoss/(double)(k+1));
		k++;
	}

	train_stream.close();
	if(a.isRoot)
		printf("save model!!\n");
	saveModel();
}

double ADMM::logloss(double p,int y){
	p = max(min(p, 1. - 1e-15), 1e-15);
	if(y == 1){
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
		g[train_ins[index]] = grad;
	}
	
	//the reduial grad
	for(uint32_t index = 0;index < numIns;index++){
		g[train_ins[index]] += rho * (x[train_ins[index]] \
			- z[train_ins[index]]) + y[train_ins[index]];
	}
}

double ADMM::predict(vector<uint32_t> train_ins)
{
	double inner = innerProduct(train_ins);
	return sigmoid(inner);
}

bool ADMM::x_update(vector<uint32_t> train_ins,int label,double *loss){
	//sgd train
	double beta = 0.01;
	
	get_Grad(train_ins,label);
	for(uint32_t index = 0;index < numIns;index++)
	{
		x[train_ins[index]] -= beta * g[train_ins[index]];
	}
	
	*loss = logloss(predict(train_ins),label);
	return true;
}

void ADMM::y_update(vector<uint32_t> train_ins){
	for(uint32_t index = 0;index < numIns;index++){
		y[train_ins[index]] += rho * (x[train_ins[index]] - z[train_ins[index]]);
	}
}

void ADMM::z_update(vector<uint32_t> train_ins){
	double s = 1.0/(rho * numProcs + 2*l2reg);
	double t = s * l1reg;

	for(uint32_t index = 0;index < numIns;index++)
	{
		w[train_ins[index]] = s * (x[train_ins[index]] + y[train_ins[index]]);
		//z_pre[train_ins[index]] = z[train_ins[index]];
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
