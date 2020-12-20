// L1-regularized logistic regression implementation using stochastic gradient descent
// (c) Tim Nugent
// timnugent@gmail.com

#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <map>
#include <omp.h>
// #include <gperftools/profiler.h>

using namespace std;

vector<string> split(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

vector<string> split(const string &s, char delim) {
    vector<string> elems;
    split(s, delim, elems);
    return elems;
}

void usage(const char* prog){

   cout << "Read training data then classify test data using logistic regression:\nUsage:\n" << prog << " [options] [training_data]" << endl << endl;
   cout << "Options:" << endl;   
   cout << "-s <int>   Shuffle dataset after each iteration. default 1" << endl;    
   cout << "-i <int>   Maximum iterations. default 50000" << endl;   
   cout << "-e <float> Convergence rate. default 0.005" << endl;    
   cout << "-a <float> Learning rate. default 0.001" << endl; 
   cout << "-l <float> L1 regularization weight. default 0.0001" << endl; 
   cout << "-m <file>  Read weights from file" << endl;    
   cout << "-o <file>  Write weights to file" << endl;   
   cout << "-t <file>  Test file to classify" << endl;     
   cout << "-p <file>  Write predictions to file" << endl;     
   cout << "-r         Randomise weights between -1 and 1, otherwise 0" << endl;    
   cout << "-v         Verbose." << endl << endl;      
}

double vecnorm(map<int,double>& w1, map<int,double>& w2){

    double sum = 0.0;
    for(auto it = w1.begin(); it != w1.end(); it++){
        double minus = w1[it->first] - w2[it->first];
        double r = minus * minus;
        sum += r;
    }
    return sqrt(sum);
}

double l1norm(map<int,double>& weights){

    double sum = 0.0;
    for(auto it = weights.begin(); it != weights.end(); it++){
        sum +=  fabs(it->second);
    }
    return sum;
}


double sigmoid(double x){

    static double overflow = 20.0;
    if (x > overflow) x = overflow;
    if (x < -overflow) x = -overflow;

    return 1.0/(1.0 + exp(-x));
}

double classify(map<int,double>& features, map<int,double>& weights){

    double logit = 0.0;
    for(auto it = features.begin(); it != features.end(); it++){
        if(it->first != 0){
            logit += it->second * weights[it->first];
        }
    }
    return sigmoid(logit);
}

map<int, double> SGD(vector<map<int,double>> data, unsigned int maxit, double l1, double alpha, map<int, double> weights, mt19937 g, map<int, double> total_l1) {
    unsigned int thread_count;
    vector<map<int, double>> sum(50);
    cout << "# stochastic gradient descent" << endl;

    #pragma omp parallel for private(weights, total_l1)
    for (auto iter = 0; iter < omp_get_num_threads(); iter++) { //data.size()
        thread_count = omp_get_num_threads();
        // Implemented Stochastic gradient descent training with cumulative L1 penalty from https://www.aclweb.org/anthology/P09-1054.pdf
        // Train(C) procedure
        double u = 0.0;
        // weights already set
        double norm = 1.0; // convergence
        // unsigned int n = 0;
        vector<int> index(data.size());
        iota(index.begin(), index.end(), 0);

        // Entries in 'data' are feature vectors
        // specifically, data[i] is a feature vector
        // there is a weight for each feature

        for (unsigned int k = 0; k <= maxit; k++) {
            // if (norm > eps) break; // cannot break with parallel loop

            // \eta = alpha, and it is non-changing
            u += (l1 * alpha);
            // save old weights before changing them
            map<int,double> old_weights(weights);
            // shuffle (randomize) vector index entries
            // if(shuf)
                shuffle(index.begin(), index.end(), g);

            // UpdateWeights(j) procedure
            // for each feature in feature vector index[i]
            for (auto it = data[index[0]].begin(); it != data[index[0]].end(); it++) {
                // since data doesn't start at 0, we ignore index 0
                if (it->first != 0) {
                    int label = data[index[0]][0];
                    // give classification of current feature vector with current weights
                    double predicted = classify(data[index[0]], weights);
                    weights[it->first] = weights[it->first] + alpha * (label - predicted) * it->second;

                    // ApplyPenalty(i) procedure
                    double z = weights[it->first];
                    if (weights[it->first] > 0.0) {
                        weights[it->first] = max(0.0, (double) (weights[it->first] - (u + total_l1[it->first])));
                    } else if(weights[it->first] < 0.0) {
                        weights[it->first] = min(0.0, (double) (weights[it->first] + (u - total_l1[it->first])));
                    }
                    total_l1[it->first] += (weights[it->first] - z);
                }
            }
            sum[omp_get_thread_num()] = weights;

            norm = vecnorm(weights, old_weights);
            if (k && k % 100 == 0) {
                double l1n = l1norm(weights);
                printf("# convergence: %1.4f l1-norm: %1.4e iterations: %i\n", norm, l1n, k);
            }
        }
    }

    map<int, double> aggregate = sum[0];
    // cout << "2: " << aggregate[1] << endl;
    for (unsigned int i = 1; i < thread_count; i++) {
        for (auto it = sum[i].begin(); it != sum[i].end(); it++)
            aggregate[it->first] += it->second;
    }
    for (auto it = aggregate.begin(); it != aggregate.end(); it++)
        aggregate[it->first] = it->second/thread_count;
    // cout << "2 again: " << aggregate[1] << endl;

    return aggregate;

    // return weights;
}

int main(int argc, const char* argv[]){
    // ProfilerStart("sgd.prof");

    // Learning rate
    double alpha = 0.001;
    // L1 penalty weight
    double l1 = 0.0001;
    // Max iterations
    unsigned int maxit = 50000;
    // Shuffle data set
    int shuf = 1;
    // Convergence threshold
    double eps = 0.005;
    // Verbose
    int verbose = 0;
    // Randomise weights
    int randw = 0;
    // Read model file
    string model_in = "";
    // Write model file
    string model_out = "";
    // Test file
    string test_file = "";   
    // Predictions file
    string predict_file = "";   

    if(argc < 2){
        usage(argv[0]);
        return(1);
    }else{
        cout << "# called with:       ";
        for(int i = 0; i < argc; i++){
            cout << argv[i] << " ";
            if(string(argv[i]) == "-a" && i < argc-1){
                alpha = atof(argv[i+1]);
            }
            if(string(argv[i]) == "-m" && i < argc-1){
                model_in = string(argv[i+1]);
            }
            if(string(argv[i]) == "-o" && i < argc-1){
                model_out = string(argv[i+1]);
            }
            if(string(argv[i]) == "-t" && i < argc-1){
                test_file = string(argv[i+1]);
            }
            if(string(argv[i]) == "-p" && i < argc-1){
                predict_file = string(argv[i+1]);
            }
            if(string(argv[i]) == "-s" && i < argc-1){
                shuf = atoi(argv[i+1]);
            }
            if(string(argv[i]) == "-i" && i < argc-1){
                maxit = atoi(argv[i+1]);
            }
            if(string(argv[i]) == "-e" && i < argc-1){
                eps = atof(argv[i+1]);
            }
            if(string(argv[i]) == "-l" && i < argc-1){
                l1 = atof(argv[i+1]);
            }
            if(string(argv[i]) == "-v"){
                verbose = 1;
            }
            if(string(argv[i]) == "-r"){
                randw = 1;
            }
            if(string(argv[i]) == "-h"){
                usage(argv[0]);
                return(1);
            }
        }
        cout << endl;
    }
 
    if(!model_in.length()){
        cout << "# learning rate:     " << alpha << endl;
        cout << "# convergence rate:  " << eps << endl;
        cout << "# l1 penalty weight: " << l1 << endl;
        cout << "# max. iterations:   " << maxit << endl;   
        cout << "# training data:     " << argv[argc-1] << endl;
        if(model_out.length()) cout << "# model output:      " << model_out << endl;
    }
    if(model_in.length()) cout << "# model input:       " << model_in << endl;
    if(test_file.length()) cout << "# test data:         " << test_file << endl;
    if(predict_file.length()) cout << "# predictions:       " << predict_file << endl;

    vector<map<int,double> > data;
    map<int,double> weights;
    map<int,double> total_l1;
    random_device rd;
    mt19937 g(rd());
    ifstream fin;
    string line;

    // Read weights from model file, if provided
    if(model_in.length()){
        fin.open(model_in.c_str());
        while (getline(fin, line)){
            if(line.length()){
                if(line[0] != '#' && line[0] != ' '){
                    vector<string> tokens = split(line,' ');
                    if(tokens.size() == 2){
                        weights[atoi(tokens[0].c_str())] = atof(tokens[1].c_str());
                    }
                }
            }
        }
        if(!weights.size()){
            cout << "# failed to read weights from file!" << endl;
            fin.close();      
            exit(-1);
        }fin.close();
    }

    // If no weights file provided, read training file and calculate weights
    if(!weights.size()){

        fin.open(argv[argc-1]);
        while (getline(fin, line)){
            if(line.length()){
                if(line[0] != '#' && line[0] != ' '){
                    vector<string> tokens = split(line,' ');
                    map<int,double> example;
                    if(atoi(tokens[0].c_str()) == 1){
                        example[0] = 1;
                    }else{
                        example[0] = 0;
                    }
                    for(unsigned int i = 1; i < tokens.size(); i++){
                        //if(strstr (tokens[i],"#") == NULL){
                            vector<string> feat_val = split(tokens[i],':');
                            if(feat_val.size() == 2){
                                example[atoi(feat_val[0].c_str())] = atof(feat_val[1].c_str());
                                if(randw){
                                    weights[atoi(feat_val[0].c_str())] = -1.0+2.0*(double)rd()/rd.max();
                                }else{
                                    weights[atoi(feat_val[0].c_str())] = 0.0;
                                }
                                total_l1[atoi(feat_val[0].c_str())] = 0.0;
                            }
                        //}
                    }
                    data.push_back(example);
                    //if(verbose) cout << "read example " << data.size() << " - found " << example.size()-1 << " features." << endl; 
                }    
            }
        }
        fin.close();

        cout << "# training examples: " << data.size() << endl;
        cout << "# features:          " << weights.size() << endl;

        // SGD begins
        double start = omp_get_wtime();
        weights = SGD(data, maxit, l1, alpha, weights, g, total_l1);
        double end = omp_get_wtime();
        cout << "# Total time for SGD: " << end - start << " seconds" << endl;

        unsigned int sparsity = 0;
        for (auto it = weights.begin(); it != weights.end(); it++) {
            if(it->second != 0) sparsity++;
        }
        printf("# sparsity:    %1.4f (%i/%i)\n",(double)sparsity/weights.size(),sparsity,(int)weights.size());     

        if(model_out.length()){
            ofstream outfile;
            outfile.open(model_out.c_str());  
            for(auto it = weights.begin(); it != weights.end(); it++){
                outfile << it->first << " " << it->second << endl;
            }
            outfile.close();
            cout << "# written weights to file " << model_out << endl;
        }

    }

    // If a test file is provided, classify it using either weights from
    // the provided weights file, or those just calculated from training
    if(test_file.length()){

        ofstream outfile;
        if(predict_file.length()){
            outfile.open(predict_file.c_str());  
        }

        cout << "# classifying" << endl;
        double tp = 0.0, fp = 0.0, tn = 0.0, fn = 0.0;
        fin.open(test_file.c_str());
        while (getline(fin, line)){
            if(line.length()){
                if(line[0] != '#' && line[0] != ' '){
                    vector<string> tokens = split(line,' ');
                    map<int,double> example;
                    int label = atoi(tokens[0].c_str());
                    for(unsigned int i = 1; i < tokens.size(); i++){
                        vector<string> feat_val = split(tokens[i],':');
                        example[atoi(feat_val[0].c_str())] = atof(feat_val[1].c_str());
                    }
                    double predicted = classify(example,weights);
                    if(verbose){
                        if(label > 0){
                            printf("label: +%i : prediction: %1.3f",label,predicted);
                        }else{
                            printf("label: %i : prediction: %1.3f",label,predicted);
                        }
                    }
                    if(predict_file.length()){
                        if(predicted >= 0.5){
                            outfile << "1" << endl;
                        }else{
                            outfile << "0" << endl;
                        }
                    }
                    if(((label == -1 || label == 0) && predicted < 0.5) || (label == 1 && predicted >= 0.5)){
                        if(label == 1){tp++;}else{tn++;}    
                        if(verbose) cout << "\tcorrect" << endl;
                    }else{
                        if(label == 1){fn++;}else{fp++;}    
                        if(verbose) cout << "\tincorrect" << endl;
                    }
                }    
            }
        }
        fin.close();

        printf ("# accuracy:    %1.4f (%i/%i)\n",((tp+tn)/(tp+tn+fp+fn)),(int)(tp+tn),(int)(tp+tn+fp+fn));
        printf ("# precision:   %1.4f\n",tp/(tp+fp));
        printf ("# recall:      %1.4f\n",tp/(tp+fn));
        printf ("# mcc:         %1.4f\n",((tp*tn)-(fp*fn))/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)));
        printf ("# tp:          %i\n",(int)tp);
        printf ("# tn:          %i\n",(int)tn);
        printf ("# fp:          %i\n",(int)fp);    
        printf ("# fn:          %i\n",(int)fn);

        if(predict_file.length()){
            cout << "# written predictions to file " << predict_file << endl;
            outfile.close();
        }    
    }

    // ProfilerStop();
    return(0);

}
