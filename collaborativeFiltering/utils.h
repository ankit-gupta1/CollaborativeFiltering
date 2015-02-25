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

#define OP_FILENAME_PRE		"..\\..\\dump\\"
#define OP_FILENAME_EXT		".txt"
#define USER_DATASET		"..\\..\\yelp_dataset_challenge_academic_dataset\\yelp_academic_dataset_user.json"
#define BUSINESS_DATASET	"..\\..\\yelp_dataset_challenge_academic_dataset\\yelp_academic_dataset_business.json"
#define REVIEW_DATASET		"..\\..\\yelp_dataset_challenge_academic_dataset\\yelp_academic_dataset_review.json"
#define BATCH_INPUT_TEXT	"input.txt"

#define LOG_FEATURES		0
#define LOG_MSE 			1

typedef enum {
	LOG_USER_FEATURES = 1,
	LOG_BUSINESS_FEATURES,
	LOG_VALIDATION_DATA,
	LOG_MEAN_SQUARE_ERROR,
} logTypes;

/* Utility to get current time and date in a string. */
string getCurrentTimeString();

/* Get the name of desired log file. */
string getLogFileName(logTypes t, unsigned int latentSpace,
		unsigned int maxIterations);

/* Log the feature vectors of business and user data. */
void logUserBusinessFeatures(double **u, double **v, unsigned int totalUsers,
		unsigned int totalBusiness,
		unsigned int latentSpace,
		unsigned int maxIterations);

/* Validate the computed features with existing reviews and log the review data. */
void validateAndLogReviews(vector<users> &allUsers,
		vector<business> &allBusiness, double **u, double **v,
		unsigned int latentSpace,
		unsigned int maxIterations);

/* Log the Mean Square Error per iteration. */
void logMsePerIteration(double *mse, unsigned int latentSpace,
		unsigned int maxIterations);

/* Compute the Mean Square Error for predicted ratings against the original ones. */
double computeMSE(vector<users> &allUsers,
		vector<business> &allBusiness, double **u, double **v,
		unsigned int latentSpace);

/* Run PMF algorithm for multiple feature space and iteration settings. */
void runPMFbatch(vector<users> &allUsers, vector<business> &allBusiness);

#endif /* UTILS_H_ */
