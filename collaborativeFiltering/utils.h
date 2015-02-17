/*
 * utils.h
 *
 *  Created on: Feb 16, 2015
 *      Author: Ankit
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <strings.h>

using namespace std;

#define OP_FILENAME_EXT		".txt"
#define USER_DATASET		"..\\..\\yelp_dataset_challenge_academic_dataset\\yelp_academic_dataset_user.json"
#define BUSINESS_DATASET	"..\\..\\yelp_dataset_challenge_academic_dataset\\yelp_academic_dataset_business.json"
#define REVIEW_DATASET		"..\\..\\yelp_dataset_challenge_academic_dataset\\yelp_academic_dataset_review.json"

string getCurrentTimeString();
void logUserBusinessFeatures(double **u, double **v, unsigned int totalUsers,
		unsigned int totalBusiness);

#endif /* UTILS_H_ */
