/*
 * utils.h
 *
 *  Created on: Feb 16, 2015
 *      Author: Ankit
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>
#include <strings.h>
#include "collab_filtering.h"

using namespace std;

/* Directory location of log dumps. */
#define OP_FILENAME_PRE		"../../dump/"

/* File extension of dump files. */
#define OP_FILENAME_EXT		".txt"

/* Filename of user data set. */
#define USER_DATASET		"../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_user.json"

/* Filename of business data set. */
#define BUSINESS_DATASET	"../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json"

/* Filename of review data. */
#define REVIEW_DATASET		"../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json"

/* Input batch file for running simulation. */
#define BATCH_INPUT_TEXT	"input.txt"

/* Netflix training data set folder. */
#define NETFLIX_DATASET		"../../netflix_dataset/training_set/"

/* Enable this macro for choosing yelp data set, default will be netflix. */
#define YELP				1

/* Number of times for which you want to cross validate. */
#define NO_OF_TRIALS		4

/* These parameters help in sampling reviews to Training : Validation : Test ::
 * 5.5 : 1 : 1, roughly speaking. With this ratio of training corpus and
 * validation data, we are preventing to over fit our model and at the same
 * time minimize the RMSE on test data as well. */
#define SAMP_PARAM_VAL		13
#define SAMP_PARAM_TEST		6

/* Enable this macro for logging collaborative filtering tuned parameters. */
#define LOG_FEATURES		0

/* Enable this macro for logging mean square error per iteration. */
#define LOG_MSE 			0

/* Enable this macro if you want to test custom models, otherwise 450 default
 * models will get tested. */
#define FORCE_INPUT			1

typedef enum {
	LOG_USER_FEATURES = 1,
	LOG_BUSINESS_FEATURES,
	LOG_TRAINING_DATA,
	LOG_VALIDATION_DATA,
	LOG_TEST_DATA,
	LOG_MEAN_SQUARE_ERROR,
	LOG_BATCH_RESULTS,
} logTypes;

/* Utility to get current time and date in a string. */
string getCurrentTimeString();

/* Get the name of desired log file. */
string getLogFileName(logTypes t, unsigned int latentSpace,
		unsigned int maxIterations);

/* Log MSE per iteration. */
void logMsePerIteration(collaborativeFiltering &collabFilteringModel);

/* Validate the computed features with existing reviews and log the review
 * data. */
void validateAndLogReviews(collaborativeFiltering &collabFilteringModel,
		logTypes logType);

/* Log the feature vectors of business and user data. */
void logUserBusinessFeatures(collaborativeFiltering &collabFilteringModel);

/* Compute the Mean Square Error for predicted ratings against the original
 * ones. */
double computeMSE(collaborativeFiltering &collabFilteringModel,
		testDataType testingDataType);

/* Run PMF algorithm for multiple feature space and iteration settings. */
void runPmfBatch(vector<users> &allUsers, vector<business> &allBusiness);

/* Log the batch run of model results. */
void logBatchResults(collaborativeFiltering *collabFilteringModel,
		unsigned int batchSize);

/* Run probabilistic matrix factorization algorithm using parallel
 * processing. */
void runPmfBatchOMP(vector<users> &allUsers, vector<business> &allBusiness);

#endif /* UTILS_H_ */
