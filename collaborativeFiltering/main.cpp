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

#if YELP

	/* Parse Yelp data set. */
	parseYelpData(allUsers, allBusiness);

#else

	/* Parse Netflix data set. */
	parseNetflixData(allUsers, allBusiness);

#endif

	/* Perform collaborative filtering using PMF. */
#if USE_GRADIENT_DESCENT

	/* Train models using gradient descent algorithm. */
	runPmfBatchGradientDescentOMP(allUsers, allBusiness);

#else

	/* Train models using multiplicative update rule algorithm. */
	runPmfBatchOMP(allUsers, allBusiness);

#endif

	return 0;
}
