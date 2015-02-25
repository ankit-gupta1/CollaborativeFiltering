/*
 * utils.cpp
 *
 *  Created on: Feb 16, 2015
 *      Author: Ankit
 */

#include <iostream>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <sstream>
#include "utils.h"
#include "collab_filtering.h"

string getCurrentTimeString() {
	time_t rawTime;
	struct tm * timeInfo;
	char buffer[80];

	time(&rawtime);
	timeInfo = localtime(&rawTime);

	strftime(buffer, 80, "%d_%m_%Y_%I_%M_%S", timeInfo);
	string str(buffer);

	return str;
}

string getLogFileName(logTypes t, unsigned int latentSpace,
		unsigned int maxIterations) {
	string logFileName;
	ostringstream strLatentSpace;
	ostringstream strMaxIterations;

	logFileName.append(OP_FILENAME_PRE);

	switch (t) {
	case LOG_USER_FEATURES:
		logFileName.append("U_");
		break;

	case LOG_BUSINESS_FEATURES:
		logFileName.append("V_");
		break;

	case LOG_VALIDATION_DATA:
		logFileName.append("D_");
		break;

	case LOG_MEAN_SQUARE_ERROR:
		logFileName.append("M_");
		break;
	}

	logFileName.append(getCurrentTimeString());
	strLatentSpace << latentSpace;
	strMaxIterations << maxIterations;
	logFileName.append("_K_");
	logFileName.append(strLatentSpace.str());
	logFileName.append("_I_");
	logFileName.append(strMaxIterations.str());
	logFileName.append(OP_FILENAME_EXT);

	return logFileName;
}

void logUserBusinessFeatures(double **u, double **v, unsigned int totalUsers,
		unsigned int totalBusiness, unsigned int latentSpace,
		unsigned int maxIterations) {
	ofstream fout;
	string logFileName;

	/* Save the user features. */
	logFileName = getLogFileName(LOG_USER_FEATURES, latentSpace,
			maxIterations);
	fout.open(logFileName.c_str());

	for (unsigned int i = 0; i < totalUsers; i++) {
		fout << "u " << setw(6) << i << ", ";
		for (unsigned int j = 0; j < latentSpace; j++) {
			fout << setw(13) << setprecision(5) << u[i][j] << ", ";
		}

		fout << endl;
	}

	fout.close();

	/* Save the business features. */
	logFileName = getLogFileName(LOG_BUSINESS_FEATURES, latentSpace,
			maxIterations);
	fout.open(logFileName.c_str());

	for (unsigned int i = 0; i < totalBusiness; i++) {
		fout << "v " << setw(6) << i << ", ";
		for (unsigned int j = 0; j < latentSpace; j++) {
			fout << setw(13) << setprecision(5) << v[i][j] << ", ";
		}

		fout << endl;
	}

	fout.close();
}

void validateAndLogReviews(vector<users> &allUsers,
		vector<business> &allBusiness, double **u, double **v,
		unsigned int latentSpace, unsigned int maxIterations) {
	unsigned int totalUsers = allUsers.size();
	ofstream fout;
	string logFileName;

	logFileName = getLogFileName(LOG_VALIDATION_DATA, latentSpace,
			maxIterations);
	fout.open(logFileName.c_str());

	for (unsigned int i = 0; i < totalUsers; i++) {
		for (unsigned int j = 0; j < allUsers[i].businessReviewed.size(); j++) {
			unsigned int businessNumID = allUsers[i].businessReviewed[j];
			double computedRating = 0.0;
			double actualRating = allUsers[i].stars[businessNumID];

			for (unsigned int k = 0; k < latentSpace; k++) {
				computedRating += u[i][k] * v[businessNumID][k];
			}

			/* Bump up the rating by 2.5*/
			computedRating += 2.5;
			fout << setw(6) << allUsers[i].numericID << ", ";
			fout << allUsers[i].genericID << ", ";
			fout << setw(6) << allBusiness[businessNumID].numericID << ", ";
			fout << allBusiness[businessNumID].genericID << ", ";
			fout << setw(5) << setprecision(5) << actualRating << ", ";
			fout << setw(10) << setprecision(5) << computedRating << ", ";
			fout << setw(13) << setprecision(5)
					<< (actualRating - computedRating) << ", " << endl;
		}
	}

	fout.close();
}

void logMsePerIteration(double *mse, unsigned int latentSpace,
		unsigned int maxIterations) {
	ofstream fout;
	string logFileName;

	logFileName = getLogFileName(LOG_MEAN_SQUARE_ERROR, latentSpace,
			maxIterations);
	fout.open(logFileName.c_str());

	for (unsigned int i = 0; i < maxIterations; i++) {
		fout << setw(3) << i + 1 << ", " << setw(10) << setprecision(5) << mse[i]
				<< "," << endl;
	}

	fout.close();
}

double computeMSE(vector<users> &allUsers, vector<business> &allBusiness,
		double **u, double **v, unsigned int latentSpace) {
	unsigned int totalUsers = allUsers.size();
	unsigned int totalReviews = 0;
	double meanSquareError = 0.0;

	for (unsigned int i = 0; i < totalUsers; i++) {
		for (unsigned int j = 0; j < allUsers[i].businessReviewed.size(); j++) {
			unsigned int businessNumID = allUsers[i].businessReviewed[j];
			double computedRating = 0.0;
			double actualRating = allUsers[i].stars[businessNumID];

			for (unsigned int k = 0; k < latentSpace; k++) {
				computedRating += u[i][k] * v[businessNumID][k];
			}

			/* Bump up the rating by 2.5*/
			computedRating += 2.5;
			totalReviews++;
			meanSquareError += (computedRating - actualRating)
					* (computedRating - actualRating);
		}
	}

	meanSquareError = meanSquareError / totalReviews;
	return meanSquareError;
}

void runPMFbatch(vector<users> &allUsers, vector<business> &allBusiness) {
	ifstream fin;
	string batchFileName;

	fin.open(BATCH_INPUT_TEXT);
	unsigned int n;
	fin >> n;

	unsigned int *latentSpace = new unsigned int[n];
	unsigned int *maxIterations = new unsigned int[n];

	for (unsigned int i = 0; i < n; i++) {
		fin >> latentSpace[i];
		fin >> maxIterations[i];
	}

	fin.close();

	for (unsigned int i = 0; i < n; i++) {
		cout << "Running PMF No Regularization Algorithm with K = "
				<< latentSpace[i] << " for " << maxIterations[i]
				<< " iterations." << endl;
		probMatFacNoReg(allUsers, allBusiness, latentSpace[i],
				maxIterations[i]);
	}

	delete latentSpace;
	delete maxIterations;
}
