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

#define OP_FILENAME_PRE		"../../dump/"
#define OP_FILENAME_EXT		".txt"
#define USER_DATASET		"../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_user.json"
#define BUSINESS_DATASET	"../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json"
#define REVIEW_DATASET		"../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json"
#define BATCH_INPUT_TEXT	"input.txt"
#define NETFLIX_DATASET		"../../netflix_dataset/training_set/"
#define YELP				1

/* Number of times for which you want to cross validate. */
#define NO_OF_TRIALS		3

/* These parameters help in sampling reviews to Training : Validation : Test ::
 * 5.5 : 1 : 1, roughly speaking. With this ratio of training corpus and
 * validation data, we are preventing to over fit our model and at the same
 * time minimize the RMSE on test data as well. */
#define SAMP_PARAM_VAL		13
#define SAMP_PARAM_TEST		6

#define LOG_FEATURES		0
#define LOG_MSE 			0
#define FORCE_INPUT			0

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

/* Validate the computed features with existing reviews and log the review data. */
void validateAndLogReviews(collaborativeFiltering &collabFilteringModel,
		logTypes logType);

/* Log the feature vectors of business and user data. */
void logUserBusinessFeatures(collaborativeFiltering &collabFilteringModel);

/* Compute the Mean Square Error for predicted ratings against the original ones. */
double computeMSE(collaborativeFiltering &collabFilteringModel,
		testDataType testingDataType);

/* Run PMF algorithm for multiple feature space and iteration settings. */
void runPmfBatch(vector<users> &allUsers, vector<business> &allBusiness);

/* Log the batch run of model results. */
void logBatchResults(collaborativeFiltering *collabFilteringModel,
		unsigned int batchSize);

#endif /* UTILS_H_ */
