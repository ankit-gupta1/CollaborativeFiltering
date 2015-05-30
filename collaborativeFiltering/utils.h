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
#define OP_FILENAME_PRE		"../dump/"

/* File extension of dump files. */
#define OP_FILENAME_EXT		".txt"

/* Filename of user data set. */
#define USER_DATASET		"../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_user.json"

/* Filename of business data set. */
#define BUSINESS_DATASET	"../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json"

/* Filename of review data. */
#define REVIEW_DATASET		"../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json"

/* Input batch file for running simulation. */
#define BATCH_INPUT_TEXT	"input.txt"

/* NETFLIX training data set folder. */
#define NETFLIX_DATASET		"../../netflix_dataset/training_set/"

/* Enable this macro for choosing yelp data set, default will be NETFLIX. */
#define YELP						1

/* Use gradient descent algorithm if you want to optimize your model using
 * cross validation and regularization parameters. */
#define USE_GRADIENT_DESCENT		1

/* Number of times for which you want to cross validate. */
#if __MINGW32__

/* For windows machine, as maximum CPU cores are 4 only. */
#define NO_OF_TRIALS				4

/* When not using gradient descent algorithm, this will ensure parallel
 * training of up to NO_OF_THREADS models. */
#define NO_OF_THREADS				4

#elif __linux__ && YELP

/* For Linux server TESSERACT, which has maximum of 8 CPU cores. */
#define NO_OF_TRIALS				1

/* When not using gradient descent algorithm, this will ensure parallel
 * training of up to NO_OF_THREADS models. */
#define NO_OF_THREADS				1

#elif __linux__ && !YELP

/* However for NETFLIX data set, use only 4 CPU cores. */
#define NO_OF_TRIALS				4

/* When not using gradient descent algorithm, this will ensure parallel
 * training of up to NO_OF_THREADS models. */
#define NO_OF_THREADS				4

#endif

/* These parameters help in sampling reviews to Training : Validation : Test ::
 * 10 : 1 : 3, roughly speaking. With this ratio of training corpus and
 * validation data, we are preventing to over fit our model and at the same
 * time minimize the RMSE on test data as well. */

#if YELP

#define SAMP_PARAM_VAL				13
#define SAMP_PARAM_TEST				4

#else

#define SAMP_PARAM_VAL				7
#define SAMP_PARAM_TEST				5

#endif

/* Enable this macro for logging collaborative filtering tuned parameters. */
#define LOG_FEATURES				0

/* Enable this macro for logging mean square error per iteration. */
#define LOG_MSE 					0

/* Enable this macro if you want to test custom models, otherwise 450 default
 * models will get tested. */
#define FORCE_INPUT					1

/* Define the minimum latent size of model. */
#define MIN_MODEL_LATENT_SPACE		4

/* Define the maximum latent size of model. */
#define MAX_MODEL_LATENT_SPACE		50

/* Define the latent space step size. */
#define LATENT_SPACE_STEP_SIZE		1

/* Define the maximum size of regularization parameter. */
#define MAX_REGULARIZATION_PARAM	(1e+2)

/* Define the minimum size of regularization parameter. */
#define MIN_REGULARIZATION_PARAM	(1e-004)

/* Try regularization range in steps of. */
#define REGULARIZATION_STEPS		18

typedef enum {
	LOG_USER_FEATURES = 1,
	LOG_BUSINESS_FEATURES,
	LOG_TRAINING_DATA,
	LOG_VALIDATION_DATA,
	LOG_TEST_DATA,
	LOG_MEAN_SQUARE_ERROR,
	LOG_BATCH_RESULTS,
} logTypes;

/* This function returns the current time in the form of underscore separated
 * time data string, which is very useful in creating new log files for each
 * run of the program. */
string getCurrentTimeString();

/* This function returns the unique log file name for each request of output log
 * file. */
string getLogFileName(logTypes t, unsigned int latentSpace,
		unsigned int maxIterations);

/* This function logs the root mean square error per iteration. This gives us
 * pretty useful information on convergence of models. */
void logMsePerIteration(collaborativeFiltering &collabFilteringModel);

/* This function validates the collaborative filtering model with the original
 * review ratings. */
void validateAndLogReviews(collaborativeFiltering &collabFilteringModel,
		logTypes logType);

/* This function is for logging feature vectors of user and business. */
void logUserBusinessFeatures(collaborativeFiltering &collabFilteringModel);

/* This function computes the root mean square error for a given set of
 * reviews.*/
double computeMSE(collaborativeFiltering &collabFilteringModel,
		testDataType testingDataType);

/* This function is for auto editing input batch file when regularization is
 * not applied while training the model. */
void editInputBatchText();

/* This function runs the different models described in the batch file and
 * train those models using multiplicative method where we can't apply
 * regularization. */
void runPmfBatchOMP(vector<users> &allUsers, vector<business> &allBusiness);

/* This function is for auto editing input batch file when regularization is
 * applied while training the model. */
void editInputBatchTextForGradientDescent();

/* This function runs the different models described in the batch file and
 * train those models using gradient descent method. */
void runPmfBatchGradientDescentOMP(vector<users> &allUsers,
		vector<business> &allBusiness);

#endif /* UTILS_H_ */
