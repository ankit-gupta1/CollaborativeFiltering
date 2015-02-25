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

	/* Parse the review data. */
	parseUsers(allUsers, userNumID);
	parseBusiness(allBusiness, businessNumID);
	parseReview(allUsers, allBusiness, userNumID, businessNumID);

	/* Perform collaborative filtering using PMF with no regularization parameters*/
	runPMFbatch(allUsers, allBusiness);

	return 0;
}
