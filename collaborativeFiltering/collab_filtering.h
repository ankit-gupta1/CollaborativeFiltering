/*
 * collab_filtering.h
 *
 *  Created on: Feb 15, 2015
 *      Author: Ankit
 */

#ifndef COLLAB_FILTERING_H_
#define COLLAB_FILTERING_H_

#include <iostream>
#include <strings.h>
#include <vector>
#include <map>

using namespace std;

typedef enum {
	TRAINING_DATA = 1, TESTING_DATA, VALIDATION_DATA
} testDataType;

/* Structure containing users info. */
struct users {
	/* Generic string based ID of each user obtained from JSON database. */
	string genericID;

	/* Numeric ID assigned to each user. */
	unsigned int numericID;

	/* List of business reviewed by given user. */
	vector<unsigned int> businessReviewed;

	/* A hash-map containing ratings provided by user to the businesses reviewed by it. */
	map<unsigned int, double> stars;
};

/* Structure containing business info. */
struct business {
	/* Generic string based ID of each business obtained from JSON database. */
	string genericID;

	/* Numeric ID assigned to each business. */
	unsigned int numericID;

	/* List of users who reviewed given business. */
	vector<unsigned int> usersReviewed;

	/* A hash-map containing ratings provided to given business by different users. */
	map<unsigned int, double> stars;
};

/* Structure containing review information*/
struct review {
	/* User ID of the user. */
	unsigned int userId;

	/* Business ID of the business. */
	unsigned int bussinessId;

	/* Rating provided by user to the business. */
	double stars;
};

/* Structure for collaborative filtering model. */
struct collaborativeFiltering {
	/* All user data. */
	vector<users> *allUsers;

	/* All business data. */
	vector<business> *allBusiness;

	/* User data used for training collaborative filtering model. */
	vector<users> trainUsers;

	/* Business data used for training collaborative filtering model. */
	vector<business> trainBusiness;

	/* Reviews of training data. */
	vector<review> trainingReviews;

	/* Reviews set aside for validation. */
	vector<review> validationReviews;

	/* Reviews set aside for testing. */
	vector<review> testReviews;

	/* Feature matrix for users. */
	double **u;

	/* Feature matrix for businesses. */
	double **v;

	/* Regularization parameter for user space. */
	double lambdaU;

	/* Regularization parameter for business space. */
	double lambdaV;

	/* Size of latent space. */
	unsigned int latentSpace;

	/* Maximum iterations for algorithm to converge. */
	unsigned int maxIterations;

	/* Mean square error generated per iteration w.r.t training data. */
	double *msePerIteration;

	/* Mean equare error of the collaborative filtering model after convergence. */
	double meanSqaureError;

	/* Are we using regularization. */
	bool isRegEnabled;
};

/* Parses the users JSON object. */
void parseUsers(vector<users> &allUsers, map<string, unsigned int> &userNumID);

/* Parses the business JSON object. */
void parseBusiness(vector<business> &allBusiness,
		map<string, unsigned int> &businessNumID);

/* Parses the reviews JSON object and populate users and business data objects. */
void parseReview(vector<users> &allUsers, vector<business> &allBusiness,
		map<string, unsigned int> &userNumID,
		map<string, unsigned int> &businessNumID);

void parseNetflixData(vector<users> &allUsers, vector<business> &allBusiness);

/* An implementation of probablistic matrix factorization algorithm.*/
void probablisticMatrixFactorization(
		collaborativeFiltering &collabFilteringModel);

/* Initialize the collaborative filtering model. */
void initCollabFilteringModel(collaborativeFiltering &collabFilteringModel,
		vector<users> &allUsers, vector<business> &allBusiness,
		unsigned int latentSpace, unsigned int maxIterations, double lambdaU,
		double lambdaV, bool isRegEnabled);

/* De-initialize the collaborative filtering model. */
void deinitCollabFilteringModel(collaborativeFiltering &collabFilteringModel);

/* Randomly pick reviews for test and validation data set*/
void randomlyPickReviews(collaborativeFiltering &collabFilteringModel,
		testDataType testingDataType);

/* Parse the yelp dataset*/
void parseYelpData(vector<users> &allUsers, vector<business> &allBusiness);

#endif /* COLLAB_FILTERING_H_ */
