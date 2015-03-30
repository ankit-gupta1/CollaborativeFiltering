/*
 * utils.cpp
 *
 *  Created on: Feb 16, 2015
 *      Author: Ankit
 */

#include <iostream>
#include <ctime>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <sstream>
#include <omp.h>
#include "utils.h"
#include "collab_filtering.h"

string getCurrentTimeString() {

	/* Create an instance for fetching current time. */
	time_t rawTime;

	/* Pointer to fetch the obtained time data. */
	struct tm * timeInfo;

	/* Character buffer for temporary usage. */
	char buffer[80];

	/* Initialize raw time object. */
	time(&rawTime);

	/* Extract current time information. */
	timeInfo = localtime(&rawTime);

	/* Format time as string. */
	strftime(buffer, 80, "%d_%m_%Y_%I_%M_%S", timeInfo);

	/* Copy buffer to the string object. */
	string str(buffer);

	/* Return the time string. */
	return str;
}

string getLogFileName(logTypes t, unsigned int latentSpace,
		unsigned int maxIterations) {

	/* String to be populated for log file name. */
	string logFileName;

	/* An object to convert latent space integer to string. */
	ostringstream strLatentSpace;

	/* An object to convert maximum iterations integer to string. */
	ostringstream strMaxIterations;

	/* An object to convert current thread ID to string. */
	ostringstream strCurrThreadID;

	/* Append the prefix of file name, which is usually the location of the
	 * dump directory folder. */
	logFileName.append(OP_FILENAME_PRE);

	/* Assign log files distinct initials as per the requested log file type.*/
	switch (t) {

	case LOG_USER_FEATURES:
		logFileName.append("US_");
		break;

	case LOG_BUSINESS_FEATURES:
		logFileName.append("VB_");
		break;

	case LOG_TRAINING_DATA:
		logFileName.append("TR_");
		break;

	case LOG_VALIDATION_DATA:
		logFileName.append("VL_");
		break;

	case LOG_TEST_DATA:
		logFileName.append("TE_");
		break;

	case LOG_MEAN_SQUARE_ERROR:
		logFileName.append("ME_");
		break;

	case LOG_BATCH_RESULTS:
		logFileName.append("BA_");
		break;
	}

	/* Append the current time string to file name, this is a very important
	 * step as it gives the unique name every single time this function is
	 * called. */
	logFileName.append(getCurrentTimeString());

	/* Convert latent space integer to string. */
	strLatentSpace << latentSpace;

	/* Convert maximum iterations integer to string. */
	strMaxIterations << maxIterations;

	/* Convert the current thread ID integer to string. */
	strCurrThreadID << omp_get_thread_num();

	/* Avoid appending max iteration and feature space strings when log file
	 * name request is for logging batch data. */
	if (t != LOG_BATCH_RESULTS) {
		logFileName.append("_K_");
		logFileName.append(strLatentSpace.str());
		logFileName.append("_I_");
		logFileName.append(strMaxIterations.str());
		logFileName.append("_TID_");
		logFileName.append(strCurrThreadID.str());
	}

	/* Attach appropriate extension to the file name. */
	logFileName.append(OP_FILENAME_EXT);

	return logFileName;
}

void logUserBusinessFeatures(collaborativeFiltering &collabFilteringModel) {

	/* Get the user feature space. */
	double **u = collabFilteringModel.u;

	/* Get the business feature space. */
	double **v = collabFilteringModel.v;

	/* Get the feature length. */
	unsigned int latentSpace = collabFilteringModel.latentSpace;

	/* Get the maximum iterations. */
	unsigned int maxIterations = collabFilteringModel.maxIterations;

	/* Get the total number of businesses. */
	unsigned int totalBusiness = collabFilteringModel.trainBusiness.size();

	/* Get the total number of users. */
	unsigned int totalUsers = collabFilteringModel.trainUsers.size();

	/* File object for logging data. */
	ofstream fout;

	/* Log file name. */
	string logFileName;

	/* Get the log file name. */
	logFileName = getLogFileName(LOG_USER_FEATURES, latentSpace, maxIterations);

	/* Open the log file. */
	fout.open(logFileName.c_str());

	/* Log the user feature space data. */
	for (unsigned int i = 0; i < totalUsers; i++) {
		fout << "u " << setw(6) << i << ", ";
		for (unsigned int j = 0; j < latentSpace; j++) {
			fout << setw(13) << setprecision(5) << u[i][j] << ", ";
		}

		fout << endl;
	}

	/* Close the log file. */
	fout.close();

	/* Get the log file name. */
	logFileName = getLogFileName(LOG_BUSINESS_FEATURES, latentSpace,
			maxIterations);

	/* Open the log file. */
	fout.open(logFileName.c_str());

	/* Log the business feature space data. */
	for (unsigned int i = 0; i < totalBusiness; i++) {
		fout << "v " << setw(6) << i << ", ";
		for (unsigned int j = 0; j < latentSpace; j++) {
			fout << setw(13) << setprecision(5) << v[i][j] << ", ";
		}

		fout << endl;
	}

	/* Close the log file. */
	fout.close();
}

void validateAndLogReviews(collaborativeFiltering &collabFilteringModel,
		testDataType testingDataType) {

	/* Use a local pointer to access the all users. */
	vector<users> *allUsers = collabFilteringModel.allUsers;

	/* Use a local pointer to access the all businesses. */
	vector<business> *allBusiness = collabFilteringModel.allBusiness;

	/* Get a local pointer to the review vector. */
	vector<review> *reviewVec;

	/* Get the log type from testing data type. */
	logTypes logType;

	/* Check for which data set log request has been made. */
	if (testingDataType == TRAINING_DATA) {
		reviewVec = &(collabFilteringModel.trainingReviews);
		logType = LOG_TRAINING_DATA;
	} else if (testingDataType == VALIDATION_DATA) {
		reviewVec = &(collabFilteringModel.validationReviews);
		logType = LOG_VALIDATION_DATA;
	} else if (testingDataType == TESTING_DATA) {
		reviewVec = &(collabFilteringModel.testReviews);
		logType = LOG_TEST_DATA;
	} else {

		/* Else return if invalid request. */
		return;
	}

	/* Get the user feature space. */
	double **u = collabFilteringModel.u;

	/* Get the business feature space. */
	double **v = collabFilteringModel.v;

	/* Get the feature length. */
	unsigned int latentSpace = collabFilteringModel.latentSpace;

	/* Get the maximum iterations. */
	unsigned int maxIterations = collabFilteringModel.maxIterations;

	/* Get the total number of reviews. */
	unsigned int totalReviews = (*reviewVec).size();

	/* File object for logging data. */
	ofstream fout;

	/* Log file name. */
	string logFileName;

	/* Get the log file name. */
	logFileName = getLogFileName(logType, latentSpace, maxIterations);

	/* Open the log file. */
	fout.open(logFileName.c_str());

	/* Iterate over all the reviews. */
	for (unsigned int i = 0; i < totalReviews; i++) {

		/* Get the business ID. */
		unsigned int businessID = (*reviewVec)[i].bussinessId;

		/* Get the user ID. */
		unsigned int userID = (*reviewVec)[i].userId;

		/* Initialize the computed rating to 0.0. */
		double computedRating = 0.0;

		/* Get the actual rating. */
		double actualRating = (*reviewVec)[i].stars;

		/* Now accumulate the rating. */
		for (unsigned int k = 0; k < latentSpace; k++) {
			computedRating += u[userID][k] * v[businessID][k];
		}

		if (computedRating > 5) {
			computedRating = 5.0;
		} else if (computedRating < 1) {
			computedRating = 1.0;
		}

		/* Now write all the pertinent data into file. */
		fout << setw(6) << userID << ", ";
		fout << (*allUsers)[userID].genericID << ", ";
		fout << setw(6) << businessID << ", ";
		fout << (*allBusiness)[businessID].genericID << ", ";
		fout << setw(5) << setprecision(5) << actualRating << ", ";
		fout << setw(10) << setprecision(5) << computedRating << ", ";
		fout << setw(13) << setprecision(5) << (actualRating - computedRating)
				<< ", " << endl;
	}

	/* Close the file. */
	fout.close();
}

void logMsePerIteration(collaborativeFiltering &collabFilteringModel) {

	/* Get the feature length. */
	unsigned int latentSpace = collabFilteringModel.latentSpace;

	/* Get the maximum iterations. */
	unsigned int maxIterations = collabFilteringModel.maxIterations;

	/* Get the array for storing MSE per iteration. */
	double *msePerIteration = collabFilteringModel.msePerIteration;

	/* File object for logging data. */
	ofstream fout;

	/* Log file name. */
	string logFileName;

	/* Get the log file name. */
	logFileName = getLogFileName(LOG_MEAN_SQUARE_ERROR, latentSpace,
			maxIterations);

	/* Open the log file. */
	fout.open(logFileName.c_str());

	/* Write the data. */
	for (unsigned int i = 0; i < maxIterations; i++) {
		fout << setw(3) << (i + 1) << ", " << setw(10) << setprecision(5)
				<< msePerIteration[i] << ", " << endl;
	}

	/* Close log file. */
	fout.close();
}

double computeMSE(collaborativeFiltering &collabFilteringModel,
		testDataType testingDataType) {

	/* Get a local pointer to the review vector. */
	vector<review> *reviewVec;

	/* Initialize the mean square error to 0. */
	double meanSquareError = 0.0;

	/* Check on which data set mse request has been made. */
	if (testingDataType == TESTING_DATA) {
		reviewVec = &(collabFilteringModel.testReviews);
	} else if (testingDataType == VALIDATION_DATA) {
		reviewVec = &(collabFilteringModel.validationReviews);
	} else if (testingDataType == TRAINING_DATA) {
		reviewVec = &(collabFilteringModel.trainingReviews);
	} else {

		/* Else return if invalid request. */
		return meanSquareError;
	}

	/* Get the user feature space. */
	double **u = collabFilteringModel.u;

	/* Get the business feature space. */
	double **v = collabFilteringModel.v;

	/* Get the feature length. */
	unsigned int latentSpace = collabFilteringModel.latentSpace;

	/* Get the total number of reviews. */
	unsigned int totalReviews = (*reviewVec).size();

	/* Iterate over all the reviews. */
	for (unsigned int i = 0; i < totalReviews; i++) {

		/* Get the business ID for the business to be reviewed. */
		unsigned int businessID = (*reviewVec)[i].bussinessId;

		/* Get the user ID. */
		unsigned int userID = (*reviewVec)[i].userId;

		/* Initialize the computed rating as 0.0*/
		double computedRating = 0.0;

		/* Get the actual rating. */
		double actualRating = (*reviewVec)[i].stars;

		/* Now accumulate the rating. */
		for (unsigned int k = 0; k < latentSpace; k++) {
			computedRating += u[userID][k] * v[businessID][k];
		}

		if (computedRating > 5) {
			computedRating = 5.0;
		} else if (computedRating < 1) {
			computedRating = 1.0;
		}

		/* Accumulate its absolute squared value*/
		meanSquareError += (computedRating - actualRating)
				* (computedRating - actualRating);
	}

	/* Compute the mean square error. */
	meanSquareError = sqrt(meanSquareError / totalReviews);

	return meanSquareError;
}

void editInputBatchText() {

	/* File stream output object. */
	ofstream fout;

	/* Open the input batch text file. */
	fout.open(BATCH_INPUT_TEXT);

	/* In the first line, write the number of models we are assessing. */
	fout << (MAX_MODEL_LATENT_SPACE - MIN_MODEL_LATENT_SPACE + 1) << endl;

	double latentSpaceSize = MIN_MODEL_LATENT_SPACE;

	while (latentSpaceSize <= MAX_MODEL_LATENT_SPACE) {

		fout << latentSpaceSize << " ";
		fout << (latentSpaceSize < 10 ? 25 : 15) << " " << endl;

		latentSpaceSize += LATENT_SPACE_STEP_SIZE;
	}

	/* Close the output file object. */
	fout.close();
}

void runPmfBatchOMP(vector<users> &allUsers, vector<business> &allBusiness) {

	/* Switch off the dynamic thread setting. */
	omp_set_dynamic(0);

	/* Output file object. */
	ofstream fout;

	/* Input file object. */
	ifstream fin;

#if !FORCE_INPUT
	editInputBatchText();
#endif

	/* Open input batch file mentioning different models. */
	fin.open(BATCH_INPUT_TEXT);

	/* Read number of models. */
	unsigned int n;
	fin >> n;

	/* Read the model parameters. */
	unsigned int *latentSpace = new unsigned int[n];
	unsigned int *maxIterations = new unsigned int[n];

	for (unsigned int i = 0; i < n; i++) {
		fin >> latentSpace[i];
		fin >> maxIterations[i];
	}

	fin.close();

	/* Log file name. */
	string logFileName;

	/* Get the log file name. */
	logFileName = getLogFileName(LOG_BATCH_RESULTS, 0, 0);

	/* Open the log file. */
	fout.open(logFileName.c_str());

	/* Now run each of these models and test them against the validation and
	 * test data sets. */
	for (unsigned int i = 0; i < n; i = i + NO_OF_THREADS) {

		/* Declare the model. */
		collaborativeFiltering collabFilteringModel[NO_OF_THREADS];

		/* Initialize the root mean square errors to 0.*/
		double rmseTraining[NO_OF_THREADS] = { 0.0 };
		double rmseTest[NO_OF_THREADS] = { 0.0 };

		/* Train the model for some fixed number of times and accumulate the
		 * error results and log its average values. */

		/* Launch as per given thread ID. */
		for (unsigned int j = 0;
				j < ((n - i) >= NO_OF_THREADS ? NO_OF_THREADS : (n - i)); j++) {
			cout << "Running PMF Algorithm with K = " << latentSpace[i + j]
					<< " for " << maxIterations[i + j] << " iterations" << endl;
		}

		/* Set the number of threads. */
		omp_set_num_threads((n - i) >= NO_OF_THREADS ? NO_OF_THREADS : (n - i));

#pragma omp parallel
		{

			/* Get the thread ID. */
			unsigned int j = omp_get_thread_num();

			/* Initialize the model. */
			initCollabFilteringModel(collabFilteringModel[j], allUsers,
					allBusiness, latentSpace[i + j], maxIterations[i + j], 0, 0,
					true);

			/* Train the model. */
			probablisticMatrixFactorization(collabFilteringModel[j]);

#if LOG_FEATURES
			cout << "Logging the computed features." << endl;

			/* Save the estimated user and business data. */
			logUserBusinessFeatures(collabFilteringModel[j]);

			/* Validate the computed features with existing reviews on training
			 * data. */
			cout << "Validating training data prediction." << endl;
			validateAndLogReviews(collabFilteringModel[j], TRAINING_DATA);

			/* Validate the computed features with existing reviews on testing
			 * data. */
			cout << "Validating training data prediction." << endl;
			validateAndLogReviews(collabFilteringModel[j], TESTING_DATA);
#endif

#if LOG_MSE
			cout << "Logging the mean square error after every iteration."
			<< endl;

			/* Log the mean square error. */
			logMsePerIteration(collabFilteringModel[j]);
#endif

			/* Compute the error on training data set. */
			rmseTraining[j] = computeMSE(collabFilteringModel[j],
					TRAINING_DATA);

			/* Compute the error on testing data set. */
			rmseTest[j] = computeMSE(collabFilteringModel[j], TESTING_DATA);

			/* De-initialize the model. */
			deinitCollabFilteringModel(collabFilteringModel[j]);
		}

		for (unsigned int j = 0;
				j < ((n - i) >= NO_OF_THREADS ? NO_OF_THREADS : (n - i)); j++) {
			/* Log the findings along with the model specifications. */
			fout << "Model " << setw(7) << i + j + 1 << ", ";
			fout << setw(13) << setprecision(5) << latentSpace[i + j] << ", ";
			fout << setw(13) << setprecision(5) << maxIterations[i + j] << ", ";
			fout << setw(13) << setprecision(5) << rmseTraining[j] << ", ";
			fout << setw(13) << setprecision(5) << rmseTest[j] << ", ";
			fout << endl;
		}
	}

	/* Close the log file.*/
	fout.close();

	/* Switch on the dynamic thread setting. */
	omp_set_dynamic(1);

	delete latentSpace;
	delete maxIterations;
}

void editInputBatchTextForGradientDescent() {
	ofstream fout;

	/* Open the input batch text file. */
	fout.open(BATCH_INPUT_TEXT);

	/* In the first line, write the number of models we are assessing. */
	fout
			<< (MAX_MODEL_LATENT_SPACE - MIN_MODEL_LATENT_SPACE + 1)
					* (REGULARIZATION_STEPS + 1) * (REGULARIZATION_STEPS + 1)
			<< endl;

	/* Start with minimum latent space size of model. */
	double latentSpaceSize = MIN_MODEL_LATENT_SPACE;

	/* Start with minimum value of regularization parameter. */
	double regularizationParamUsers = MIN_REGULARIZATION_PARAM;
	double regularizationParamBusiness = MIN_REGULARIZATION_PARAM;

	/* Determine the step size of regularization parameters in log linear
	 * manner. */
	double regularizationStepSize = (log10(MAX_REGULARIZATION_PARAM)
			- log10(MIN_REGULARIZATION_PARAM)) / REGULARIZATION_STEPS;

	/* Now generate different model hyper parameters by covering all the models
	 * and combinations of different regularization parameters of users and
	 * business. */
	while (latentSpaceSize <= MAX_MODEL_LATENT_SPACE) {
		while (regularizationParamUsers <= MAX_REGULARIZATION_PARAM) {
			while (regularizationParamBusiness <= MAX_REGULARIZATION_PARAM) {
				fout << latentSpaceSize << " ";
				fout << 1000 << " ";
				fout << regularizationParamUsers << " ";
				fout << regularizationParamBusiness << endl;

				regularizationParamBusiness = pow(10,
						log10(regularizationParamBusiness)
								+ regularizationStepSize);
			}

			regularizationParamUsers = pow(10,
					log10(regularizationParamUsers) + regularizationStepSize);
			regularizationParamBusiness = MIN_REGULARIZATION_PARAM;
		}

		latentSpaceSize += LATENT_SPACE_STEP_SIZE;
		regularizationParamUsers = MIN_REGULARIZATION_PARAM;
		regularizationParamBusiness = MIN_REGULARIZATION_PARAM;
	}

	/* Close the output file object. */
	fout.close();
}

void runPmfBatchGradientDescentOMP(vector<users> &allUsers,
		vector<business> &allBusiness) {

	/* Switch off the dynamic thread setting. */
	omp_set_dynamic(0);

	/* Set the number of threads. */
	omp_set_num_threads(NO_OF_TRIALS);

	/* Output file object. */
	ofstream fout;

	/* Input file object. */
	ifstream fin;

#if !FORCE_INPUT
	editInputBatchTextForGradientDescent();
#endif

	/* Open input batch file mentioning different models. */
	fin.open(BATCH_INPUT_TEXT);

	/* Read number of models. */
	unsigned int n;
	fin >> n;

	/* Read the model parameters. */
	unsigned int *latentSpace = new unsigned int[n];
	unsigned int *maxIterations = new unsigned int[n];
	double *lambdaU = new double[n];
	double *lambdaV = new double[n];

	for (unsigned int i = 0; i < n; i++) {
		fin >> latentSpace[i];
		fin >> maxIterations[i];
		fin >> lambdaU[i];
		fin >> lambdaV[i];
	}

	fin.close();

	/* Log file name. */
	string logFileName;

	/* Get the log file name. */
	logFileName = getLogFileName(LOG_BATCH_RESULTS, 0, 0);

	/* Open the log file. */
	fout.open(logFileName.c_str());

	/* Now run each of these models and test them against the validation and
	 * test data sets. */
	for (unsigned int i = 0; i < n; i++) {

		/* Declare the model. */
		collaborativeFiltering collabFilteringModel[NO_OF_TRIALS];

		/* Initialize the root mean square errors to 0.*/
		double rmseTraining[NO_OF_TRIALS] = { 0.0 };
		double rmseValidation[NO_OF_TRIALS] = { 0.0 };
		double rmseTest[NO_OF_TRIALS] = { 0.0 };
		double rmsTrainError = 0.0;
		double rmsValidationError = 0.0;
		double rmsTestError = 0.0;

		/* Train the model for some fixed number of times and accumulate the
		 * error results and log its average values. */

		/* Launch as per given thread ID. */
		cout << "Running PMF Algorithm with K = " << latentSpace[i] << " for "
				<< maxIterations[i] << " iterations, lambda U = " << lambdaU[i]
				<< ", lambda V = " << lambdaV[i] << endl << endl;

		collaborativeFiltering collabFilteringTest;
		/* Initialize the model. */
		initCollabFilteringModel(collabFilteringTest, allUsers, allBusiness,
				latentSpace[0], maxIterations[0], lambdaU[0], lambdaV[0], true);

		vector<review> testReviews = collabFilteringTest.testReviews;
		vector<users> trainUsers = collabFilteringTest.trainUsers;
		vector<business> trainBusiness = collabFilteringTest.trainBusiness;

		deinitCollabFilteringModel(collabFilteringTest);

#pragma omp parallel
		{

			/* Get the thread ID. */
			unsigned int j = omp_get_thread_num();

			/* Initialize the model. */
			collabFilteringModel[j].testReviews = testReviews;
			initCollabFilteringModel(collabFilteringModel[j], trainUsers,
					trainBusiness, latentSpace[i], maxIterations[i], lambdaU[i],
					lambdaV[i], false);

			/* Train the model. */
			probablisticMatrixFactorizationGradientDescent(
					collabFilteringModel[j]);

#if LOG_FEATURES
			cout << "Logging the computed features." << endl;

			/* Save the estimated user and business data. */
			logUserBusinessFeatures(collabFilteringModel[j]);

			cout << "Validating the computed features." << endl;

			/* Validate the computed features with existing reviews on training
			 * data. */
			validateAndLogReviews(collabFilteringModel[j], TRAINING_DATA);

			/* Validate the computed features with existing reviews on
			 * validation data. */
			validateAndLogReviews(collabFilteringModel[j], VALIDATION_DATA);

			/* Validate the computed features with existing reviews on testing
			 * data. */
			validateAndLogReviews(collabFilteringModel[j], TESTING_DATA);
#endif

#if LOG_MSE
			cout << "Logging the mean square error after every iteration."
			<< endl;

			/* Log the mean square error. */
			logMsePerIteration(collabFilteringModel[j]);
#endif

			/* Compute the error on training data set. */
			rmseTraining[j] = computeMSE(collabFilteringModel[j],
					TRAINING_DATA);

			/* Compute the error on validation data set. */
			rmseValidation[j] = computeMSE(collabFilteringModel[j],
					VALIDATION_DATA);

			/* Compute the error on testing data set. */
			rmseTest[j] = computeMSE(collabFilteringModel[j], TESTING_DATA);

			/* De-initialize the model. */
			deinitCollabFilteringModel(collabFilteringModel[j]);
		}

		for (unsigned int j = 0; j < NO_OF_TRIALS; j++) {
			rmsTestError += rmseTest[j];
			rmsValidationError += rmseValidation[j];
			rmsTrainError += rmseTraining[j];
		}

		/* Log the findings along with the model specifications. */
		fout << "Model " << setw(7) << i + 1 << ", ";
		fout << setw(13) << setprecision(5) << latentSpace[i] << ", ";
		fout << setw(13) << setprecision(5) << maxIterations[i] << ", ";
		fout << setw(13) << setprecision(5) << lambdaU[i] << ", ";
		fout << setw(13) << setprecision(5) << lambdaV[i] << ", ";
		fout << setw(13) << setprecision(5) << rmsTrainError / NO_OF_TRIALS
				<< ", ";
		fout << setw(13) << setprecision(5) << rmsValidationError / NO_OF_TRIALS
				<< ", ";
		fout << setw(13) << setprecision(5) << rmsTestError / NO_OF_TRIALS
				<< ", ";
		fout << endl;
	}

	/* Close the log file.*/
	fout.close();

	/* Switch on the dynamic thread setting. */
	omp_set_dynamic(1);

	delete latentSpace;
	delete maxIterations;
	delete lambdaU;
	delete lambdaV;
}
