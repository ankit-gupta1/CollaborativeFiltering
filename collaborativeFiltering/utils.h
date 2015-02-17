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

#define OP_FILENAME_EXT		".txt"
#define USER_DATASET		"..\\..\\yelp_dataset_challenge_academic_dataset\\yelp_academic_dataset_user.json"
#define BUSINESS_DATASET	"..\\..\\yelp_dataset_challenge_academic_dataset\\yelp_academic_dataset_business.json"
#define REVIEW_DATASET		"..\\..\\yelp_dataset_challenge_academic_dataset\\yelp_academic_dataset_review.json"

/* Utility to get current time and date in a string. */
string getCurrentTimeString();

/* Log the feature vectors of business and user data. */
void logUserBusinessFeatures(double **u, double **v, unsigned int totalUsers,
		unsigned int totalBusiness);

/* Validate the computed features with existing reviews and log the review data.*/
void validateAndLogReviews(vector<users> &allUsers,
		vector<business> &allBusiness, double **u, double **v);

#endif /* UTILS_H_ */
