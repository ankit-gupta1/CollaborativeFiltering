/*
 * data_parsing.cpp
 *
 *  Created on: Feb 15, 2015
 *      Author: Ankit
 */

#include <iostream>
#include <fstream>
#include "rapidjson/document.h"     
#include "rapidjson/prettywriter.h"
#include "rapidjson/filestream.h"
#include "collab_filtering.h"
#include "utils.h"
#include <dirent.h>
#include <set>
#include <map>

using namespace std;
using namespace rapidjson;

void parseUsers(vector<users> &allUsers, map<string, unsigned int> &userNumID) {

	/* Input file stream object. */
	ifstream ifs(USER_DATASET);

	/* String for reading each line of the file. */
	string str;

	/* Rapid JSON object used for parsing JSON objects from text. */
	Document document;

	/* Counter to keep track of number of users. */
	unsigned int count = 0;

	cout << "Parsing user data" << endl;

	/* Iterate till the file is read completely. */
	while (getline(ifs, str)) {

		/* Create a new user. */
		users user;

		/* Create a data buffer. */
		char *buffer = (char *) str.c_str();

		/* Parse the document and verify if there is any error in the parsing
		 * process. */
		if (document.ParseInsitu(buffer).HasParseError()) {
			return;
		}

		/* Fetch the encrypted user string ID from the parsed JSON object. */
		user.genericID = document["user_id"].GetString();

		/* Assign a numeric ID to the user object. */
		user.numericID = count++;

		/* Also populate the hash map keeping a track of user encrypted string
		 * name versus its numeric ID. */
		userNumID[user.genericID] = user.numericID;

		/* Push the user created in the universal list. */
		allUsers.push_back(user);
	}
}

void parseBusiness(vector<business> &allBusiness,
		map<string, unsigned int> &businessNumID) {

	/* Input file stream object. */
	ifstream ifs(BUSINESS_DATASET);

	/* String for reading each line of the file. */
	string str;

	/* Rapid JSON object used for parsing JSON objects from text. */
	Document document;

	/* Counter to keep track of number of users. */
	unsigned int count = 0;

	cout << "Parsing business data" << endl;

	/* Iterate till the file is read completely. */
	while (getline(ifs, str)) {

		/* Create the business object. */
		business busines;

		/* Create a data buffer. */
		char *buffer = (char *) str.c_str();

		/* Parse the document and verify if there is any error in the parsing
		 * process. */
		if (document.ParseInsitu(buffer).HasParseError()) {
			return;
		}

		/* Fetch the encrypted business string ID from the parsed JSON
		 * object. */
		busines.genericID = document["business_id"].GetString();

		/* Assign a numeric ID to the business. */
		busines.numericID = count++;

		/* Populate the hash map for maintaining a one to one correspondence
		 * between business numeric ID and its encrypted string ID. */
		businessNumID[busines.genericID] = busines.numericID;

		/* Push the user created in the universal list. */
		allBusiness.push_back(busines);
	}
}

void parseReview(vector<users> &allUsers, vector<business> &allBusiness,
		map<string, unsigned int> &userNumID,
		map<string, unsigned int> &businessNumID) {

	/* Input file stream object. */
	ifstream ifs(REVIEW_DATASET);

	/* String for reading each line of the file. */
	string str;

	/* Rapid JSON object used for parsing JSON objects from text. */
	Document document;

	cout << "Parsing review data" << endl;

	/* Read each line of the file till it gets empty. */
	while (getline(ifs, str)) {

		/* Pointer to point the business object. */
		business *busines;

		/* Pointer to point the user object. */
		users *user;

		/* Create a data buffer. */
		char *buffer = (char *) str.c_str();

		/* Parse the document and verify if there is any error in the parsing
		 * process. */
		if (document.ParseInsitu(buffer).HasParseError()) {
			return;
		}

		/* Get the user numeric ID from the string to numeric ID hash map. */
		unsigned int userNumericID = userNumID[document["user_id"].GetString()];

		/* Get the business numeric ID from the string to numeric ID hash
		 * map. */
		unsigned int businessNumericID =
				businessNumID[document["business_id"].GetString()];

		/* Get the pointer to that corresponding user. */
		user = &allUsers[userNumericID];

		/* Get the pointer to that corresponding business. */
		busines = &allBusiness[businessNumericID];

		/* Aggregate the numeric ID of business reviewed by the user. */
		user->businessReviewed.push_back(businessNumericID);

		/* Store the ratings provided by that user to the business in a hash
		 * map. */
		user->stars[businessNumericID] = document["stars"].GetDouble();

		/* Aggregate the numeric ID of user which reviewed this business. */
		busines->usersReviewed.push_back(userNumericID);

		/* Store the ratings given to that business by user in a hash map. */
		busines->stars[userNumericID] = document["stars"].GetDouble();
	}
}

void parseNetflixData(vector<users> &allUsers, vector<business> &allBusiness) {
	/* Pointer for locating the data-set directory. */
	DIR *dir;

	/* Pointer for each of the filenames present in the directory. */
	struct dirent *ent;

	/* Map containing the one to one mapping of each user ID to its
	 * user count value when they were discovered. */
	map<string, users> userMap;

	/* Map to keep track of numeric ID of each user with respect to
	 * their generic string IDs*/
	map<unsigned int, string> userMapNum;

	/* A counter to keep track of number of users. */
	unsigned int userCount = 0;

	/* A counter to keep track of number of businesses. */
	unsigned int businessCount = 0;

	/* Open the directory. */
	dir = opendir(NETFLIX_DATASET);

	/* Loop in the files till all the data files have been parsed. */
	while ((ent = readdir(dir)) != NULL) {

		/* Get the directory name. */
		string reviewFileName = NETFLIX_DATASET;

		/* Get the name of the files. */
		reviewFileName.append(ent->d_name);

		/* Open the file. */
		ifstream fin;
		fin.open(reviewFileName.c_str());

		/* Parse only if the file is valid.*/
		if (fin && reviewFileName.find("mv") != std::string::npos) {

			/* Get the first string of the filename. It will and must contain
			 * the movie index. */
			string str;
			getline(fin, str);
			business currentBusiness;

			/* Populate the generic ID and numeric ID of the movie. */
			currentBusiness.genericID = str.substr(0, 1);
			currentBusiness.numericID = businessCount++;

			/* Print the number of movies parsed as of now. */
			cout << "Parsing movie # " << businessCount << endl;

			/* Read the file till end of file. */
			while (!fin.eof()) {

				/* Read the first line. */
				if (getline(fin, str)) {

					/* Find the first comma location. It will provide us the
					 * numeric information about user who rated the movie. */
					unsigned int commaLocation;
					commaLocation = str.find(',', 0);

					/* Get the numerical string till that comma location.*/
					string userID = str.substr(0, commaLocation);

					/* A pointer to the current user. */
					users *currentUser;

					/* Verify if the current user has been created earlier. */
					if (userMap.find(userID) != userMap.end()) {

						/* If yes, then get the user information from the hash
						 * map. */
						currentUser = &(userMap[userID]);
					} else {

						/* Create a new user and assign the generic and numeric
						 * IDs. */
						currentUser = new users;
						currentUser->numericID = userCount++;
						currentUser->genericID = userID;
					}

					/* Get the rating given by user for that movie. */
					double rating = atoi(&str[commaLocation + 1]);

					/* Put the movie ID in business reviewed list. */
					currentUser->businessReviewed.push_back(
							currentBusiness.numericID);

					/* Populate the ratings hash map of user. */
					currentUser->stars[currentBusiness.numericID] = rating;

					/* Put the numeric business ID in the users reviewed
					 * list. */
					currentBusiness.usersReviewed.push_back(
							currentUser->numericID);

					/* Populate the ratings hash map of the movie. */
					currentBusiness.stars[currentUser->numericID] = rating;

					/* Add the user to the hash map, if not added earlier.*/
					if (userMap.find(userID) == userMap.end()) {

						/* Add user to the hash map*/
						userMap[userID] = *currentUser;
						userMapNum[userCount - 1] = userID;
						delete currentUser;
					}
				}
			}

			/* Save the current movie in the list. */
			allBusiness.push_back(currentBusiness);
		}

		fin.close();
	}

	/* Close the file directory*/
	cout << "User count " << userCount << endl;
	cout << "Business count " << businessCount << endl;

	/* Now reorder all the users in the hash map, put them into a list as per
	 * their numeric ID. */
	for (unsigned int i = 0; i < userCount; i++) {

		/* Get the iterator to current location of i in hash map. */
		map<unsigned int, string>::iterator itr1 = userMapNum.find(i);

		/* Now get the object corresponding to that location. */
		map<string, users>::iterator itr2 = userMap.find(itr1->second);

		/* Push the user object into the vector. */
		allUsers.push_back(itr2->second);

		/* Start erasing redundant data. */
		userMap.erase(itr2);

		/* Start erasing redundant data. */
		userMapNum.erase(itr1);
	}

	closedir(dir);
}

void parseYelpData(vector<users> &allUsers, vector<business> &allBusiness) {
	/* A hash-map for user's generic string ID to its assigned numeric ID. */
	map<string, unsigned int> userNumID;

	/* A hash-map for business's generic string ID to its assigned
	 * numeric ID. */
	map<string, unsigned int> businessNumID;

	/* Parse the user data. */
	parseUsers(allUsers, userNumID);

	/* Parse the business data. */
	parseBusiness(allBusiness, businessNumID);

	/* Parse the review data. */
	parseReview(allUsers, allBusiness, userNumID, businessNumID);
}
