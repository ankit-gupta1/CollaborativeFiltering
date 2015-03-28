/*
 * collab_filtering.cpp
 *
 *  Created on: Feb 23, 2015
 *      Author: Ankit
 */

#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <set>
#include "collab_filtering.h"
#include "utils.h"
#include <cstring>
#include <random>

/* This static counter helps in generating random seeds so that sampling of
 * review data for validation and testing purposes must be as random as
 * possible. This further helps in generalization of model. */
static int counter = 1;

void randomlyPickReviews(collaborativeFiltering &collabFilteringModel,
		testDataType testingDataType) {

	/* Use a local pointer to access the training users. */
	vector<users> *trainUsers = &(collabFilteringModel.trainUsers);

	/* Use a local pointer to access the training businesses. */
	vector<business> *trainBusiness = &(collabFilteringModel.trainBusiness);

	/* Use a local pointer to access the appropriate review set
	 * requested to get populated. */
	vector<review> *reviewVec;
	unsigned int sampling_parameter;

	/* Pick the review data set as per the request. */
	if (testingDataType == VALIDATION_DATA) {
		reviewVec = &(collabFilteringModel.validationReviews);
		sampling_parameter = SAMP_PARAM_VAL;
	} else if (testingDataType == TESTING_DATA) {
		reviewVec = &(collabFilteringModel.testReviews);
		sampling_parameter = SAMP_PARAM_TEST;
	} else {

		/* Else return from the function. */
		return;
	}

	/* Assign access locations. */
	vector<unsigned int> accessLocation;
	for (unsigned int i = 0; i < (*trainUsers).size(); i++) {
		accessLocation.push_back(i);
	}

	/* Now randomize the access locations. */
	srand(time(NULL) + counter++ * 7074);
	random_shuffle(accessLocation.begin(), accessLocation.end());

	/* Iterate through all the users and verify if some of the reviews
	 * can be picked up for the given review set. */
	for (unsigned int itrUsers = 0; itrUsers < (*trainUsers).size();
			itrUsers++) {

		/* Get some random user. */
		unsigned int userID = accessLocation[itrUsers];

		/* Number of reviews to be picked up. */
		unsigned int targetReviews = 0;

		/* A counter tracing if target reviews picking is completed. */
		unsigned int count = 0;

		/* Get the number of reviews to be picked up. */
		unsigned int numOfBusinessReviewed =
				(*trainUsers)[userID].businessReviewed.size();

		/* Review picking rule, if the number of businesses reviewed by
		 * the user is less than 5, then pick only one review, else pick
		 * at most 20% of the reviews. */
		if (numOfBusinessReviewed > 1
				&& numOfBusinessReviewed <= sampling_parameter) {
			targetReviews = 1;
		} else if (numOfBusinessReviewed > sampling_parameter) {
			targetReviews = (numOfBusinessReviewed / sampling_parameter);
		}

		/* A counter to iterate over all the businesses reviewed by the
		 * user. */
		unsigned int businessCount = 0;

		/* Randomly shuffle the list of businesses reviewed. Shuffling here
		 * will do no harm and will help in picking as random reviews as
		 * possible. */
		srand(time(NULL) + counter++ * 19002);
		random_shuffle((*trainUsers)[userID].businessReviewed.begin(),
				(*trainUsers)[userID].businessReviewed.end());

		/* While loop used here for iterating over all businesses with the
		 * condition to break once the target reviews are done reviewing.
		 * Also make sure that after every iteration, number of business
		 * reviews left with user are more than 1. */
		while (count < targetReviews && businessCount < numOfBusinessReviewed
				&& numOfBusinessReviewed > 1) {

			/* Get the business ID of the business reviewed. */
			unsigned int businessID =
					(*trainUsers)[userID].businessReviewed[businessCount];

			/* Now verify if the business to be removed from train set has been
			 * reviewed by multiple users. If not, then this business ID cannot
			 * be removed. */
			if ((*trainBusiness)[businessID].usersReviewed.size() > 1) {

				/* Create a new review. */
				review newReview;

				/* Populate the user ID. */
				newReview.userId = userID;

				/* Populate the business ID. */
				newReview.bussinessId = businessID;

				/* Get the rating awarded by user 'i' to business
				 * 'businessID'. */
				newReview.stars = (*trainUsers)[userID].stars[businessID];

				/* Push the review to the review vector. */
				(*reviewVec).push_back(newReview);

				/* Now remove this review from user and business lists.*/
				/* Get the iterator of this business location. */
				vector<unsigned int>::iterator itr;
				map<unsigned int, double>::iterator itr_map;
				itr = find((*trainUsers)[userID].businessReviewed.begin(),
						(*trainUsers)[userID].businessReviewed.end(),
						businessID);

				/* Erase the businessID from the list of business. */
				if (itr != (*trainUsers)[userID].businessReviewed.end()) {
					(*trainUsers)[userID].businessReviewed.erase(itr);
				}

				/* Erase the businessID from the map as well. */
				itr_map = (*trainUsers)[userID].stars.find(businessID);
				if (itr_map != (*trainUsers)[userID].stars.end()) {
					(*trainUsers)[userID].stars.erase(itr_map);
				}

				/* Now update the size of number of businesses. */
				numOfBusinessReviewed =
						(*trainUsers)[userID].businessReviewed.size();

				/* Fetch the iterator of this user location in the
				 * corresponding business object. */
				itr = find((*trainBusiness)[businessID].usersReviewed.begin(),
						(*trainBusiness)[businessID].usersReviewed.end(),
						userID);

				/* Erase the user ID from the list of users who reviewed this
				 * business. */
				if (itr != (*trainBusiness)[businessID].usersReviewed.end()) {
					(*trainBusiness)[businessID].usersReviewed.erase(itr);
				}

				/* Erase the userID from its corresponding map as well. */
				itr_map = (*trainBusiness)[businessID].stars.find(userID);
				if (itr_map != (*trainBusiness)[businessID].stars.end()) {
					(*trainBusiness)[businessID].stars.erase(itr_map);
				}

				/* Now update the counter used to track the target
				 * reviews to be picked. */
				count++;
			} else {

				/* Note the business counter value should not be increased
				 * if we are erasing the businessId from the list of
				 * businesses reviewed. Otherwise we will be skipping a
				 * potential business to get picked. */
				businessCount++;
			}
		}
	}
}

void initCollabFilteringModel(collaborativeFiltering &collabFilteringModel,
		vector<users> &allUsers, vector<business> &allBusiness,
		unsigned int latentSpace, unsigned int maxIterations, double lambdaU,
		double lambdaV, bool isRegEnabled) {

	/* Get the user feature space. */
	double ***u = &(collabFilteringModel.u);

	/* Get the business feature space. */
	double ***v = &(collabFilteringModel.v);

	/* Get the array for storing MSE per iteration*/
	double **msePerIteration = &(collabFilteringModel.msePerIteration);

	/* Copy the original all user data to training data set*/
	collabFilteringModel.trainUsers = allUsers;
	collabFilteringModel.trainBusiness = allBusiness;

	/* Use a local pointer to access the training users. */
	vector<users> *trainUsers = &(collabFilteringModel.trainUsers);

	/* Use a local pointer to access the training businesses. */
	vector<business> *trainBusiness = &(collabFilteringModel.trainBusiness);

	/* Also keep the pointer to original data set. It will be
	 * useful for computing the mean square error. */
	collabFilteringModel.allUsers = &(allUsers);
	collabFilteringModel.allBusiness = &(allBusiness);

	/* If regularization is enabled then we have to create a
	 * validation and test set. */
	if (isRegEnabled) {

		/* Pick the validation data set. */
		randomlyPickReviews(collabFilteringModel, VALIDATION_DATA);

		/* Pick the test data set. */
		randomlyPickReviews(collabFilteringModel, TESTING_DATA);
	} else {

		/* Pick the test data set. */
		randomlyPickReviews(collabFilteringModel, TESTING_DATA);
	}

	/* Populate training reviews of whatever left after picking
	 * validation and test set reviews. Iterate through all the
	 * users. */
	for (unsigned int i = 0; i < (*trainUsers).size(); i++) {

		/* Iterate through the hash map of user. */
		for (map<unsigned int, double>::iterator itr =
				(*trainUsers)[i].stars.begin();
				itr != (*trainUsers)[i].stars.end(); itr++) {

			/* Create a new review object. */
			review newReview;

			/* Assign the user ID i. */
			newReview.userId = i;

			/* Get the key to set as the business ID. */
			newReview.bussinessId = itr->first;

			/* Get its corresponding value which will be the rating
			 * given by user 'i' to this business.*/
			newReview.stars = itr->second;

			/* Now push the review to review vector. */
			collabFilteringModel.trainingReviews.push_back(newReview);
		}
	}

#if 0
	cout << "Number of training data points "
	<< collabFilteringModel.trainingReviews.size() << endl;
	cout << "Number of test data points "
	<< collabFilteringModel.testReviews.size() << endl;
	if (isRegEnabled) {
		cout << "Number of validation data points "
		<< collabFilteringModel.validationReviews.size() << endl;
	}
#endif

	/* Populate the latent space or feature length variable. */
	collabFilteringModel.latentSpace = latentSpace;

	/* Set the maximum iterations. */
	collabFilteringModel.maxIterations = maxIterations;

	/* Set the regularization parameter for users. */
	collabFilteringModel.lambdaU = lambdaU;

	/* Set the regularization parameter for businesses. */
	collabFilteringModel.lambdaV = lambdaV;

	/* Set the regularization enabled flag. */
	collabFilteringModel.isRegEnabled = isRegEnabled;

	/* Now allocate memory to the user feature vector. */
	*u = new double*[(*trainUsers).size()];
	for (unsigned int i = 0; i < (*trainUsers).size(); i++) {
		(*u)[i] = new double[latentSpace];
	}

	/* Now allocate memory to the business feature vector. */
	*v = new double*[(*trainBusiness).size()];
	for (unsigned int i = 0; i < (*trainBusiness).size(); i++) {
		(*v)[i] = new double[latentSpace];
	}

	/* Initialize user variables to zero. */
	for (unsigned int i = 0; i < (*trainUsers).size(); i++) {
		for (unsigned int j = 0; j < latentSpace; j++) {
			(*u)[i][j] = 0;
		}
	}

	default_random_engine generator;
	uniform_real_distribution<double> distribution(0, 1);

	/* Randomly initialize business variables. */
	for (unsigned int i = 0; i < (*trainBusiness).size(); i++) {
		for (unsigned int j = 0; j < latentSpace; j++) {
			(*v)[i][j] = distribution(generator);
		}
	}

	/* Initialize MSE per iteration. */
	*msePerIteration = new double[maxIterations];
	memset((void *) *msePerIteration, 0, sizeof(double) * maxIterations);
}

void deinitCollabFilteringModel(collaborativeFiltering &collabFilteringModel) {

	/* Use a local pointer to access the training users. */
	vector<users> *trainUsers = &(collabFilteringModel.trainUsers);

	/* Use a local pointer to access the training businesses. */
	vector<business> *trainBusiness = &(collabFilteringModel.trainBusiness);

	/* Get the user feature space. */
	double ***u = &(collabFilteringModel.u);

	/* Get the business feature space. */
	double ***v = &(collabFilteringModel.v);

	/* Get the array for storing MSE per iteration*/
	double **msePerIteration = &(collabFilteringModel.msePerIteration);

	/* Free up the allocated memory. */
	for (unsigned int i = 0; i < (*trainUsers).size(); i++) {
		delete (*u)[i];
	}

	delete *u;

	for (unsigned int i = 0; i < (*trainBusiness).size(); i++) {
		delete (*v)[i];
	}

	delete *v;
	delete *msePerIteration;

	/* Erase all the vectors. */
	collabFilteringModel.trainBusiness.clear();
	collabFilteringModel.trainUsers.clear();
	collabFilteringModel.validationReviews.clear();
	collabFilteringModel.testReviews.clear();
	collabFilteringModel.trainingReviews.clear();

	/* Decouple the pointers to all users and business objects. */
	collabFilteringModel.allUsers = NULL;
	collabFilteringModel.allBusiness = NULL;
}

void probablisticMatrixFactorization(
		collaborativeFiltering &collabFilteringModel) {

	/* Use a local pointer to access the training users. */
	vector<users> *trainUsers = &(collabFilteringModel.trainUsers);

	/* Use a local pointer to access the training businesses. */
	vector<business> *trainBusiness = &(collabFilteringModel.trainBusiness);

	/* Get the user feature space. */
	double **u = collabFilteringModel.u;

	/* Get the business feature space. */
	double **v = collabFilteringModel.v;

	/* Get the total number of businesses. */
	unsigned int totalBusiness = collabFilteringModel.trainBusiness.size();

	/* Get the total number of users. */
	unsigned int totalUsers = collabFilteringModel.trainUsers.size();

	/* Get the feature length. */
	unsigned int latentSpace = collabFilteringModel.latentSpace;

	/* Get the maximum iterations. */
	unsigned int maxIterations = collabFilteringModel.maxIterations;

	/* Main algorithm starts here. Here probablistic matrix
	 * factorization is implemented. Algorithm will run for
	 * fixed number of preconfigured iterations. */
	for (unsigned int k = 0; k < maxIterations; k++) {

		/* First update the feature values of all the users. */
		for (unsigned int i = 0; i < totalUsers; i++) {
			double *temp = new double[latentSpace];
			double den = 0.0;

			/* Update using only those business features, which user has rated.
			 * Iterate over all the businesses reviewed by the user. */
			memset((void *) temp, 0, sizeof(double) * latentSpace);
			for (unsigned int j = 0;
					j < (*trainUsers)[i].businessReviewed.size(); j++) {

				/* Get the numeric businessID of the business reviewed. */
				unsigned int businessID = (*trainUsers)[i].businessReviewed[j];

				/* Get the corresponding rating given by user to this
				 * business. */
				double z_ij = (*trainUsers)[i].stars[businessID];

				/* Start accumulating user features as per update rule. */
				for (unsigned int l = 0; l < latentSpace; l++) {
					temp[l] += z_ij * v[businessID][l];
					den += v[businessID][l] * v[businessID][l];
				}
			}

			/* Update user features. */
			den = den - collabFilteringModel.lambdaU;
			for (unsigned int j = 0; j < latentSpace; j++) {
				u[i][j] = temp[j] / den;
			}

			delete temp;
		}

		/* Secondly update the feature values of all the businesses. */
		for (unsigned int i = 0; i < totalBusiness; i++) {
			double *temp = new double[latentSpace];
			double den = 0.0;

			/* Update using only those user ratings, which have reviewed the
			 * business. Iterate over all the users who reviewed the current
			 * business. */
			memset((void *) temp, 0, sizeof(double) * latentSpace);
			for (unsigned int j = 0;
					j < (*trainBusiness)[i].usersReviewed.size(); j++) {

				/* Get the numeric useID of the user. */
				unsigned int userID = (*trainBusiness)[i].usersReviewed[j];

				/* Get the corresponding rating received by business from this
				 * user. */
				double z_ij = (*trainBusiness)[i].stars[userID];

				/* Start accumulating business features as per update rule. */
				for (unsigned int l = 0; l < latentSpace; l++) {
					temp[l] += z_ij * u[userID][l];
					den += u[userID][l] * u[userID][l];
				}
			}

			/* Update business features. */
			den = den - collabFilteringModel.lambdaV;
			for (unsigned int j = 0; j < latentSpace; j++) {
				v[i][j] = temp[j] / den;
			}

			delete temp;
		}

#if LOG_MSE
		/* Compute the mean square error per iteration. */
		collabFilteringModel.msePerIteration[k] =
		computeMSE(collabFilteringModel, TRAINING_DATA);
#endif
	}
}
