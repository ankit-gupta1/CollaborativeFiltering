/*
 * main.cpp
 *
 *  Created on: Feb 15, 2015
 *      Author: Ankit
 */

#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include "collab_filtering.h"
#include "utils.h"

using namespace std;

int main() {
	/* List of all the users present in the database. */
	vector<users> allUsers;

	/* List of all the business present in the database. */
	vector<business> allBusiness;

	/* A hash-map for user's generic string ID to its assigned numeric ID. */
	map<string, unsigned int> userNumID;

	/* A hash-map for business's generic string ID to its assigned numeric ID. */
	map<string, unsigned int> businessNumID;

	unsigned int totalUsers;
	unsigned int totalBusiness;

	/* Parse the review data.*/
	parseUsers(allUsers, userNumID);
	parseBusiness(allBusiness, businessNumID);
	parseReview(allUsers, allBusiness, userNumID, businessNumID);

	totalUsers = allUsers.size();
	totalBusiness = allBusiness.size();

	cout << totalUsers << endl;
	cout << totalBusiness << endl;

	/* Feature variable of each user of size LATENT_SPACE. */
	double **u;

	/* Feature variable of each business of size LATENT_SPACE. */
	double **v;

	/* Assign memory to variables. */
	u = new double*[totalUsers];
	v = new double*[totalBusiness];

	for (unsigned int i = 0; i < totalUsers; i++) {
		u[i] = new double[LATENT_SPACE];
	}

	for (unsigned int i = 0; i < totalBusiness; i++) {
		v[i] = new double[LATENT_SPACE];
	}

	/* Randomly initialize business variables. */
	for (unsigned int i = 0; i < totalBusiness; i++) {
		for (unsigned int j = 0; j < LATENT_SPACE; j++) {
			srand(time(NULL) + (i + 1) * (j + 1) + 78 + i * j);
			v[i][j] = (double(rand() % 101) / double(20)) - 2.5;
		}
	}

	/* Initialize user variables to zero. */
	for (unsigned int i = 0; i < totalUsers; i++) {
		for (unsigned int j = 0; j < LATENT_SPACE; j++) {
			u[i][j] = 0;
		}
	}

	/*
	 * Main algorithm starts here.
	 * Here probablistic matrix factorization is implemented.
	 * Algorithm will run for fixed number of pre-configured
	 * iterations.
	 */
	for (unsigned int k = 0; k < MAX_ITERATION; k++) {

		/* First update the feature values of all the users. */
		for (unsigned int i = 0; i < totalUsers; i++) {
			double temp[LATENT_SPACE] = { 0.0 };
			double den = 0.0;

			/*
			 * Update using only those business features, which user has rated.
			 * Iterate over all the businesses reviewed by the user.
			 */
			for (unsigned int j = 0; j < allUsers[i].businessReviewed.size();
					j++) {

				/* Get the numeric businessID of the business reviewed. */
				unsigned int businessID = allUsers[i].businessReviewed[j];

				/* Get the corresponding rating given by user to this business. */
				double z_ij = allUsers[i].stars[businessID] - 2.5;

				/* Start accumulating user features as per update rule. */
				for (unsigned int l = 0; l < LATENT_SPACE; l++) {
					temp[l] += z_ij * v[businessID][l];
					den += v[businessID][l] * v[businessID][l];
				}
			}

			/* Update user features. */
			for (unsigned int j = 0; j < LATENT_SPACE; j++) {
				u[i][j] = temp[j] / den;
			}
		}

		/* Secondly update the feature values of all the businesses. */
		for (unsigned int i = 0; i < totalBusiness; i++) {
			double temp[LATENT_SPACE] = { 0.0 };
			double den = 0.0;

			/*
			 * Update using only those user ratings, which have reviwed the business.
			 * Iterate over all the users who reviwed the current business.
			 */
			for (unsigned int j = 0; j < allBusiness[i].usersReviewed.size();
					j++) {

				/* Get the numeric useID of the user. */
				unsigned int userID = allBusiness[i].usersReviewed[j];

				/* Get the corresponding rating received by business from this user. */
				double z_ij = allBusiness[i].stars[userID] - 2.5;

				/* Start accumulating business features as per update rule. */
				for (unsigned int l = 0; l < LATENT_SPACE; l++) {
					temp[l] += z_ij * u[userID][l];
					den += u[userID][l] * u[userID][l];
				}
			}

			/* Update business features. */
			for (unsigned int j = 0; j < LATENT_SPACE; j++) {
				v[i][j] = temp[j] / den;
			}
		}

		cout << "Iteration " << k + 1 << " completed!" << endl;
	}

	cout << "Logging the computed features." << endl;
	/* Save the estimated user and business data. */
	logUserBusinessFeatures(u, v, totalUsers, totalBusiness);

	cout << "Validating the computed features." << endl;
	/* Validate the computed features with existing reviews. */
	validateAndLogReviews(allUsers, allBusiness, u, v);

	/* Free up the allocated memory. */
	for (unsigned int i = 0; i < totalUsers; i++) {
		delete u[i];
	}
	delete u;

	for (unsigned int i = 0; i < totalBusiness; i++) {
		delete v[i];
	}
	delete v;

	return 0;
}
