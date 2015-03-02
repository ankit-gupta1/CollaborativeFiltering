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
#include "utils.h"
#include "collab_filtering.h"

/* This look up table contains the number of iterations required by each model
 * to achieve a sub-optimal root mean square error on the training data. We
 * would like to call it as the sweet spot because if we aim to over optimize
 * our model on the training data set, it badly suffers from the phenomena of
 * over-fitting. Also, these iteration caps for each model were found after
 * experimenting on fixed iterations of 25 for different feature lengths
 * ranging from 3 to 20. */
const unsigned int sweetSpotOptIter[18] { 22, 16, 13, 12, 12, 12, 11, 11, 11, 10,
		10, 10, 10, 9, 9, 9, 9, 9, };

/* This function returns the current time in the form of underscore separated
 * time data string, which is very useful in creating new log files for each
 * run of the program. */
string getCurrentTimeString() {
	time_t rawTime;
	struct tm * timeInfo;
	char buffer[80];

	/* Initialize raw time object. */
	time(&rawTime);

	/* Extract current time information. */
	timeInfo = localtime(&rawTime);

	/* Format time as string. */
	strftime(buffer, 80, "%d_%m_%Y_%I_%M_%S", timeInfo);

	/* Copy buffer to the string object. */
	string str(buffer);

	return str;
}

/* This function returns the unique log file name for each request of output log
 * file. */
string getLogFileName(logTypes t, unsigned int latentSpace,
		unsigned int maxIterations) {

	/* String to be populated for log file name. */
	string logFileName;

	/* An object to convert latent space integer to string. */
	ostringstream strLatentSpace;

	/* An object to convert maximum iterations integer to string. */
	ostringstream strMaxIterations;

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

	/* Extract strings from integers. */
	strLatentSpace << latentSpace;
	strMaxIterations << maxIterations;

	/* Avoid appending max iteration and feature space strings when log file
	 * name request is for logging batch data. */
	if (t != LOG_BATCH_RESULTS) {
		logFileName.append("_K_");
		logFileName.append(strLatentSpace.str());
		logFileName.append("_I_");
		logFileName.append(strMaxIterations.str());
	}

	/* Attach appropriate extension to the file name. */
	logFileName.append(OP_FILENAME_EXT);

	return logFileName;
}

/* This function is for logging feature vectors of user and business. */
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

/* This function validates the collaborative filtering model with the original
 * review ratings. */
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
	}
	else {

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
		unsigned int businessNumID = (*reviewVec)[i].bussNumId;

		/* Get the user ID. */
		unsigned int userNumID = (*reviewVec)[i].userNumId;

		/* Initialize the computed rating to 0.0. */
		double computedRating = 0.0;

		/* Get the actual rating. */
		double actualRating = (*reviewVec)[i].stars;

		/* Now accumulate the rating. */
		for (unsigned int k = 0; k < latentSpace; k++) {
			computedRating += u[userNumID][k] * v[businessNumID][k];
		}

		/* Bump up the rating by 2.5*/
		computedRating += 2.5;

		/* Now write all the pertinent data into file. */
		fout << setw(6) << userNumID << ", ";
		fout << (*allUsers)[userNumID].genericID << ", ";
		fout << setw(6) << businessNumID << ", ";
		fout << (*allBusiness)[businessNumID].genericID << ", ";
		fout << setw(5) << setprecision(5) << actualRating << ", ";
		fout << setw(10) << setprecision(5) << computedRating << ", ";
		fout << setw(13) << setprecision(5) << (actualRating - computedRating)
				<< ", " << endl;
	}

	/* Close the file. */
	fout.close();
}

/* This function logs the root mean square error per iteration. This gives us
 * pretty useful information on convergence of models. */
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

/* This function computes the root mean square error for a given set of
 * reviews.*/
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
		unsigned int businessNumID = (*reviewVec)[i].bussNumId;

		/* Get the user ID. */
		unsigned int userID = (*reviewVec)[i].userNumId;

		/* Initialize the computed rating as 0.0*/
		double computedRating = 0.0;

		/* Get the actual rating. */
		double actualRating = (*reviewVec)[i].stars;

		/* Now accumulate the rating. */
		for (unsigned int k = 0; k < latentSpace; k++) {
			computedRating += u[userID][k] * v[businessNumID][k];
		}

		/* Bump up the rating by 2.5*/
		computedRating += 2.5;

		/* Accumulate its absolute squared value*/
		meanSquareError += (computedRating - actualRating)
				* (computedRating - actualRating);
	}

	/* Compute the mean square error. */
	meanSquareError = sqrt(meanSquareError / totalReviews);
	return meanSquareError;
}

/* This function is for running different collaborative filtering models in
 * batch mode. This is very useful for cross validating and fine tuning the
 * collaborative filtering model. In our implementation, the basic tuning
 * parameters of filtering model are feature space length, the number of
 * iterations required for model to attain a sub-optimal value and the
 * regularization parameters for feature matrices of user and business. */
void runPmfBatch(vector<users> &allUsers, vector<business> &allBusiness) {
	/* Output file object. */
	ofstream fout;

	/* Input file object. */
	ifstream fin;

#if !FORCE_INPUT
	/* Open the input batch text file. */
	fout.open(BATCH_INPUT_TEXT);

	/* In the first line, write the number of models we are assessing. */
	fout << 18 * 25 << endl;

	/* This is followed by mentioning the feature or latent space length, max
	 * iterations, a boolean variable mentioning if regularization is used or
	 * not, regularization parameter for user and business variables. */
	for (unsigned int i = 3; i < 21; i++) {
		for (unsigned int m = 2; m < 11; m = m + 2) {
			for (unsigned int n = 2; n < 11; n = n + 2) {
				fout << i << " ";
				fout << sweetSpotOptIter[i - 3] << " ";
				fout << 1 << " ";
				fout << double(double(m) / double(1000000)) << " ";
				fout << double(double(n) / double(1000000)) << endl;
			}
		}
	}

	/* Close the output file object. */
	fout.close();
#endif

	/* Open input batch file mentioning different models. */
	fin.open(BATCH_INPUT_TEXT);

	/* Read number of models. */
	unsigned int n;
	fin >> n;

	/* Read the model parameters. */
	unsigned int *latentSpace = new unsigned int[n];
	unsigned int *maxIterations = new unsigned int[n];
	unsigned int *isRegEnabled = new unsigned int[n];
	double *lambdaU = new double[n];
	double *lambdaV = new double[n];

	for (unsigned int i = 0; i < n; i++) {
		fin >> latentSpace[i];
		fin >> maxIterations[i];
		fin >> isRegEnabled[i];
		if (isRegEnabled[i] == 1) {
			fin >> lambdaU[i];
			fin >> lambdaV[i];
		} else {
			lambdaU[i] = 0.0;
			lambdaV[i] = 0.0;
		}
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
		collaborativeFiltering collabFilteringModel;

		/* Initialize the root mean square errors to 0.*/
		double rmseTraining = 0.0;
		double rmseValidation = 0.0;
		double rmseTest = 0.0;

		/* Train the model for some fixed number of times and accumulate the
		 * error results and log its average values. */
		for (unsigned int j = 0; j < NO_OF_TRIALS; j++) {

			cout << "Running PMF Algorithm with K = " << latentSpace[i] << " for "
					<< maxIterations[i] << " iterations, lambda U = " << lambdaU[i]
					<< ", lambda V = " << lambdaV[i] << ", trial " << j + 1 <<  endl;

			/* Initialize the model. */
			initCollabFilteringModel(collabFilteringModel, allUsers,
					allBusiness, latentSpace[i], maxIterations[i], lambdaU[i],
					lambdaV[i], (isRegEnabled[i] == 1) ? true : false);

			/* Train the model. */
			probablisticMatrixFactorization(collabFilteringModel);

#if LOG_FEATURES
			cout << "Logging the computed features." << endl;

			/* Save the estimated user and business data. */
			logUserBusinessFeatures(collabFilteringModel);

			cout << "Validating the computed features." << endl;

			/* Validate the computed features with existing reviews on training
			 * data. */
			validateAndLogReviews(collabFilteringModel, TRAINING_DATA);

			/* Validate the computed features with existing reviews on
			 * validation data. */
			validateAndLogReviews(collabFilteringModel, VALIDATION_DATA);

			/* Validate the computed features with existing reviews on testing
			 * data. */
			validateAndLogReviews(collabFilteringModel, TESTING_DATA);
#endif

#if LOG_MSE
			cout << "Logging the mean square error after every iteration."
					<< endl;

			/* Log the mean square error. */
			logMsePerIteration(collabFilteringModel);
#endif

			/* Compute the error on training data set. */
			rmseTraining += computeMSE(collabFilteringModel, TRAINING_DATA);

			/* Compute the error on validation data set. */
			rmseValidation += computeMSE(collabFilteringModel, VALIDATION_DATA);

			/* Compute the error on testing data set. */
			rmseTest += computeMSE(collabFilteringModel, TESTING_DATA);

			/* De-initialize the model. */
			deinitCollabFilteringModel(collabFilteringModel);
		}

		/* Log the findings along with the model specifications. */
		fout << "Model " << setw(7) << i + 1 << ", ";
		fout << setw(13) << setprecision(5) << latentSpace[i] << ", ";
		fout << setw(13) << setprecision(5) << maxIterations[i] << ", ";
		fout << setw(13) << setprecision(5) << lambdaU[i] << ", ";
		fout << setw(13) << setprecision(5) << lambdaV[i] << ", ";
		fout << setw(13) << setprecision(5) << rmseTraining / NO_OF_TRIALS << ", ";
		fout << setw(13) << setprecision(5) << rmseValidation / NO_OF_TRIALS << ", ";
		fout << setw(13) << setprecision(5) << rmseTest / NO_OF_TRIALS << ", ";
		fout << endl;
	}

	/* Close the log file.*/
	fout.close();

	delete latentSpace;
	delete maxIterations;
	delete isRegEnabled;
	delete lambdaU;
	delete lambdaV;
}
