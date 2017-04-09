#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>

#include "conf_util.h"
#include "str_util.h"

using namespace std;
using namespace util;

static double sigmoid(double inx)
{
	return 1.0 / (1 + exp(-inx));
}

static double predict_ins(double *model,vector<uint32_t> ins)
{
	double innerp = 0.0;
	vector<uint32_t>::const_iterator iter = ins.begin();
	
	while(iter != ins.end())
	{
		innerp += model[*iter];
		iter++;
	}
	
	return sigmoid(innerp);
}

int main(int argc,char **argv)
{
	if(argc < 2){
		cout << "Usage: ./predict conf_file" << endl;
		exit(0);
	}
	
	string conf_file(argv[1]);
	conf_util admm_conf;
	admm_conf.parse(conf_file);
	
	string valid_file = admm_conf.getItem<string>(string("valid_file"));
	string model_file = admm_conf.getItem<string>(string("model_file"));
	string out_file = admm_conf.getItem<string>(string("out_file"));
	uint32_t num_fea = admm_conf.getItem<uint32_t>(string("num_fea"));
	
	double *model = new double[num_fea];
	
	ifstream valid_stream(valid_file);
	ifstream model_stream(model_file);
	ofstream out_stream(out_file);
	
	if(valid_stream.fail()){
		cout << "can not open valid file!" << endl;
		return 0;
	}
	
	if(model_stream.fail()){
		cout << "can not open model file" << endl;
		return 0;
	}
	
	for(uint32_t index = 0;index < num_fea;index++)
	{
		model_stream >> model[index];
	}
	
	cout << "begin to predict..." << endl;
	string line;
	double score;
	while(getline(valid_stream,line))
	{
		vector<string> fields;
		str_util::split(line," ",fields);
		
		vector<uint32_t> ins;
		ins.clear();
		
		for(uint32_t i = 1;i < fields.size() - 1;i++)
		{
			ins.push_back((uint32_t)atoi(fields[i+1].c_str()));
		}
		score = predict_ins(model,ins);
		out_stream << score << " " << line << endl;
	}
	
	valid_stream.close();
	out_stream.close();
	model_stream.close();

	cout << "finish predict..." << endl;
		
	return 0;
}
