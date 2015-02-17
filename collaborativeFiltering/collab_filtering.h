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

/* Size of the latent space of each of the users and business objects. */
#define LATENT_SPACE	50

/* Set maximum iterations. */
#define MAX_ITERATION	500

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

/* Parses the users JSON object. */
void parseUsers(vector<users> &allUsers, map<string, unsigned int> &userNumID);

/* Parses the business JSON object. */
void parseBusiness(vector<business> &allBusiness,
		map<string, unsigned int> &businessNumID);

/* Parses the reviews JSON object and populate users and business data objects. */
void parseReview(vector<users> &allUsers, vector<business> &allBusiness,
		map<string, unsigned int> &userNumID,
		map<string, unsigned int> &businessNumID);

#endif /* COLLAB_FILTERING_H_ */
