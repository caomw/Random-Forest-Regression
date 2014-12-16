//
//  main.cpp
//  RandomForestRegression
//
//  Created by jimmy on 12/9/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#include <iostream>
#include <vector>
#include "rf_rgrsn_constant_regressor.h"
#include "rf_rgrsn_constant_regressor_builder.h"
#include "ff_rgrsn_regressor.h"
#include "ff_rgrsn_builder.h"

using namespace std;

// read/write data
bool read_feature_labels(const char *file, const int feature_dim, vector<nl_vector> & features, vector<double> & labels);
bool read_sin_noise_3(vector<double> & gds, vector<double> & noisy_gds);
void generateMultiScaleFeature(const vector<double> & data,
                               const vector<double> & groundTruth,
                               const vector<unsigned int> & windowSize,
                               const int step,
                               vector< vector<nl_vector > > & features,
                               vector<double> & labels);


// example code
void test_constant_regressor();
void test_filter_forest_regressor();

bool read_feature_labels(const char *file, const int feature_dim, vector<nl_vector> & features, vector<double> & labels)
{
    FILE *pf = fopen(file, "r");
    if (!pf) {
        printf("Error: can not read from %s\n", file);
        return false;
    }
    int num = 0;
    int dim = 0;
    int ret = fscanf(pf, "%d %d", &num, &dim);
    assert(ret == 2);
    assert(dim == feature_dim + 1);
    
    nl_vector d(feature_dim, 0);
    for (int i = 0; i<num; i++) {
        for (int j = 0; j<feature_dim; j++) {
            ret = fscanf(pf, "%lf ", &d[j]);
            assert(ret == 1);
        }
        features.push_back(d);
        
        double label = 0;
        ret = fscanf(pf, "%lf", &label);
        assert(ret == 1);
        labels.push_back(label);
    }
    assert(features.size() == labels.size());
    fclose(pf);
    return true;
}

bool read_sin_noise_3(vector<double> & gds, vector<double> & noisy_gds, const char *file)
{
    assert(file);
    FILE *pf = fopen(file, "r");
    if (!pf) {
        printf("can not open %s\n", file);
        return false;
    }
    while (1) {
        double gd = 0;
        double noisy_gd = 0;
        int ret = fscanf(pf, "%lf %lf ", &gd, &noisy_gd);
        if (ret != 2) {
            break;
        }
        gds.push_back(gd);
        noisy_gds.push_back(noisy_gd);
    }
    fclose(pf);
    assert(gds.size() == noisy_gds.size());
    printf("load %d data\n", (int)gds.size());
    return true;
}

void generateMultiScaleFeature(const vector<double> & data,
                               const vector<double> & groundTruth,
                               const vector<unsigned int> & windowSize,
                               const int step,
                               vector< vector<nl_vector > > & features,
                               vector<double> & labels)
{
    assert(step >= 1);
    assert(data.size() == groundTruth.size());
    
    const int beingInd = windowSize.back();
    for (int i = beingInd; i<data.size(); i++) {
        vector<nl_vector > multi_scale_feature;
        // loop all windowSize
        for (int j = 0; j<windowSize.size(); j++) {
            const int feat_length = windowSize[j]/step;
            assert(feat_length > 0);
            nl_vector feat(feat_length, 0);
            for (int k = 0; k<feat_length; k += 1) {
                feat[k] = data[i - k * step];
            }
            multi_scale_feature.push_back(feat);
        }
        features.push_back(multi_scale_feature);
        labels.push_back(groundTruth[i]);
    }
    
    assert(features.size() == labels.size());
}



void test_constant_regressor()
{
    // change directory when run in different machine
    string train_file("/Users/jimmy/Desktop/images/cpsc540_final_project/data_2D_train.txt");
    
    int feature_dim = 2;
    vector<nl_vector> train_features;
    vector<double> train_labels;
    read_feature_labels(train_file.c_str(), feature_dim, train_features, train_labels);
    
    rf_rgrsn_constant_regressor_builder builder;
    
    unsigned int tree_num = 10;
    unsigned int max_depth = 6;
    unsigned int min_node_size = 6;
    unsigned int min_sample_num = 5;
    
    rf_rgrsn_tree_cost_function_constant costFunction;
    rf_rgrsn_constant_regressor *pRegressor = builder.new_regressioner();
    
    builder.set_tree_number(tree_num);
    builder.set_tree_depth(max_depth);
    builder.set_min_node_size(min_node_size);
    builder.set_min_sample_num(min_sample_num);
    
    builder.build_model(*pRegressor, train_features, train_labels, costFunction);
    
    string regressorFile("temp_regressor_.txt");
    pRegressor->write(regressorFile.c_str());
    
    pRegressor = builder.new_regressioner();
    pRegressor->read(regressorFile.c_str());
    
    // testing data
    string test_file("/Users/jimmy/Desktop/images/cpsc540_final_project/data_2D_test.txt");
    vector<nl_vector> test_features;
    vector<double> test_labels;
    read_feature_labels(test_file.c_str(), feature_dim, test_features, test_labels);
    
    double test_rms = 0.0;
    vector<double> predicted_labels;
    for (int i = 0; i<test_features.size(); i++) {
        double val = pRegressor->regression(test_features[i]);
        predicted_labels.push_back(val);
        test_rms += (val - test_labels[i]) * (val - test_labels[i]);
    }
    test_rms = sqrt(test_rms/predicted_labels.size());
    printf("test rms is %f\n", test_rms);
}

void test_filter_forest_regressor()
{
    string training_file("/Users/jimmy/Desktop/images/simulating_data/sin_gd_noise_07.txt");
    string test_file("/Users/jimmy/Desktop/images/simulating_data/testing_sin_gd_noise_07.txt");

    vector<double> train_gd;
    vector<double> train_observer;
    bool isRead = read_sin_noise_3(train_gd, train_observer, training_file.c_str());
    assert(isRead);
    
    vector<unsigned int> windowSizes;
    windowSizes.push_back(3);
    windowSizes.push_back(5);
    //windowSizes.push_back(6);
    const int step = 1;
    
    vector<vector<nl_vector > > multi_scale_features;
    vector<double> labels;
    generateMultiScaleFeature(train_observer, train_gd, windowSizes, step, multi_scale_features, labels);
    
    vector<double> test_gd;
    vector<double> test_observer;
    read_sin_noise_3(test_gd, test_observer, test_file.c_str());
    
    vector<vector<nl_vector > > test_multi_scale_features;
    vector<double> test_labels;
    generateMultiScaleFeature(train_observer, train_gd, windowSizes, step, test_multi_scale_features, test_labels);
    
    // training
    ff_rgrsn_builder builder;
    ff_rgrsn_regressor *pRegressor = builder.new_regressioner();
    
    ff_tree_cost_function costFunction;
    builder.set_min_node_size(12);
    builder.set_tree_depth(6);
    builder.set_tree_number(20);
    
    builder.build_model(*pRegressor, multi_scale_features, labels, costFunction);
    
    //
    string save_file("ff_regressor_temp_.txt");
    pRegressor->write(save_file.c_str());
    
    pRegressor = builder.new_regressioner();
    pRegressor->read(save_file.c_str());
    
    // testing
    vector<double> testResult;
    double rms = 0;
    for (int i = 0; i<test_multi_scale_features.size(); i++) {
        double val = pRegressor->regression(test_multi_scale_features[i]);
        testResult.push_back(val);
        double dif = val - test_labels[i];
        rms += dif * dif;
    }
    rms = sqrt(rms/test_labels.size());
    printf("test rms is %f\n", rms);
    
}

#if 1
int main(int argc, const char * argv[])
{
    // test constant

    test_constant_regressor();
    test_filter_forest_regressor();
    
    return 0;
}
#endif

