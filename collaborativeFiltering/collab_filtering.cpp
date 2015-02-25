/*
 * collab_filtering.cpp
 *
 *  Created on: Feb 23, 2015
 *      Author: Ankit
 */

#include <ctime>
#include <cstdlib>
#include "collab_filtering.h"
#include "utils.h"

void probMatFacNoReg(vector<users> &allUsers, vector<business> &allBusiness,
		unsigned int latentSpace, unsigned int maxIterations) {
	unsigned int totalBusiness = allBusiness.size();
	unsigned int totalUsers = allUsers.size();
	double *msePerIteration = new double[maxIterations];

	/* Feature variable of each user of size latentSpace. */
	double **u;

	/* Feature variable of each business of size latentSpace. */
	double **v;

	/* Assign memory to variables. */
	u = new double*[totalUsers];
	v = new double*[totalBusiness];

	for (unsigned int i = 0; i < totalUsers; i++) {
		u[i] = new double[latentSpace];
	}

	for (unsigned int i = 0; i < totalBusiness; i++) {
		v[i] = new double[latentSpace];
	}

	/* Randomly initialize business variables. */
	for (unsigned int i = 0; i < totalBusiness; i++) {
		for (unsigned int j = 0; j < latentSpace; j++) {
			srand(time(NULL) + (i + 1) * (j + 1) + 78 + i * j);
			v[i][j] = (double(rand() % 101) / double(20)) - 2.5;
		}
	}

	/* Initialize user variables to zero. */
	for (unsigned int i = 0; i < totalUsers; i++) {
		for (unsigned int j = 0; j < latentSpace; j++) {
			u[i][j] = 0;
		}
	}

	/*
	 * Main algorithm starts here.
	 * Here probablistic matrix factorization is implemented.
	 * Algorithm will run for fixed number of pre-configured
	 * iterations.
	 */
	for (unsigned int k = 0; k < maxIterations; k++) {
		/* First update the feature values of all the users. */
		for (unsigned int i = 0; i < totalUsers; i++) {
			double *temp = new double[latentSpace];
			double den = 0.0;

			/*
			 * Update using only those business features, which user has rated.
			 * Iterate over all the businesses reviewed by the user.
			 */
			memset((void *) temp, 0, sizeof(double) * latentSpace);
			for (unsigned int j = 0; j < allUsers[i].businessReviewed.size();
					j++) {

				/* Get the numeric businessID of the business reviewed. */
				unsigned int businessID = allUsers[i].businessReviewed[j];

				/* Get the corresponding rating given by user to this business. */
				double z_ij = allUsers[i].stars[businessID] - 2.5;

				/* Start accumulating user features as per update rule. */
				for (unsigned int l = 0; l < latentSpace; l++) {
					temp[l] += z_ij * v[businessID][l];
					den += v[businessID][l] * v[businessID][l];
				}
			}

			/* Update user features. */
			for (unsigned int j = 0; j < latentSpace; j++) {
				u[i][j] = temp[j] / den;
			}

			delete temp;
		}

		/* Secondly update the feature values of all the businesses. */
		for (unsigned int i = 0; i < totalBusiness; i++) {
			double *temp = new double[latentSpace];
			double den = 0.0;

			/*
			 * Update using only those user ratings, which have reviwed the business.
			 * Iterate over all the users who reviwed the current business.
			 */
			memset((void *) temp, 0, sizeof(double) * latentSpace);
			for (unsigned int j = 0; j < allBusiness[i].usersReviewed.size();
					j++) {

				/* Get the numeric useID of the user. */
				unsigned int userID = allBusiness[i].usersReviewed[j];

				/* Get the corresponding rating received by business from this user. */
				double z_ij = allBusiness[i].stars[userID] - 2.5;

				/* Start accumulating business features as per update rule. */
				for (unsigned int l = 0; l < latentSpace; l++) {
					temp[l] += z_ij * u[userID][l];
					den += u[userID][l] * u[userID][l];
				}
			}

			/* Update business features. */
			for (unsigned int j = 0; j < latentSpace; j++) {
				v[i][j] = temp[j] / den;
			}

			delete temp;
		}

		cout << "Iteration " << k + 1 << " completed!" << endl;

#if LOG_MSE
		msePerIteration[k] = computeMSE(allUsers, allBusiness, u, v,
				latentSpace);
#endif
	}

#if LOG_FEATURES
	cout << "Logging the computed features." << endl;

	/* Save the estimated user and business data. */
	logUserBusinessFeatures(u, v, totalUsers, totalBusiness, latentSpace,
			maxIterations);

	cout << "Validating the computed features." << endl;

	/* Validate the computed features with existing reviews. */
	validateAndLogReviews(allUsers, allBusiness, u, v, latentSpace,
			maxIterations);
#endif

#if LOG_MSE
	cout << "Logging the mean square error after every iteration." << endl;

	/* Log the mean square error. */
	logMsePerIteration(msePerIteration, latentSpace, maxIterations);
#endif

	/* Free up the allocated memory. */
	for (unsigned int i = 0; i < totalUsers; i++) {
		delete u[i];
	}

	delete u;

	for (unsigned int i = 0; i < totalBusiness; i++) {
		delete v[i];
	}

	delete v;
	delete msePerIteration;
}
