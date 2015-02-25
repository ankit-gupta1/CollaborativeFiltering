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

using namespace std;
using namespace rapidjson;

void parseUsers(vector<users> &allUsers, map<string, unsigned int> &userNumID) {
	ifstream ifs(USER_DATASET);
	string str;
	Document document;
	unsigned int count = 0;

	cout << "Parsing user data" << endl;

	while (getline(ifs, str)) {
		users user;
		char *buffer = (char *) str.c_str();

		if (document.ParseInsitu(buffer).HasParseError()) {
			return;
		}

		user.genericID = document["user_id"].GetString();
		user.numericID = count++;
		userNumID[user.genericID] = user.numericID;
		allUsers.push_back(user);
	}
}

void parseBusiness(vector<business> &allBusiness,
		map<string, unsigned int> &businessNumID) {
	ifstream ifs(BUSINESS_DATASET);
	string str;
	Document document;
	unsigned int count = 0;

	cout << "Parsing business data" << endl;

	while (getline(ifs, str)) {
		business busines;
		char *buffer = (char *) str.c_str();

		if (document.ParseInsitu(buffer).HasParseError()) {
			return;
		}

		busines.genericID = document["business_id"].GetString();
		busines.numericID = count++;
		businessNumID[busines.genericID] = busines.numericID;
		allBusiness.push_back(busines);
	}
}

void parseReview(vector<users> &allUsers, vector<business> &allBusiness,
		map<string, unsigned int> &userNumID,
		map<string, unsigned int> &businessNumID) {
	ifstream ifs(REVIEW_DATASET);
	string str;
	Document document;

	cout << "Parsing review data" << endl;

	while (getline(ifs, str)) {
		business *busines;
		users *user;
		char *buffer = (char *) str.c_str();

		if (document.ParseInsitu(buffer).HasParseError()) {
			return;
		}

		unsigned int userNumericID;
		unsigned int businessNumericID;

		userNumericID = userNumID[document["user_id"].GetString()];
		businessNumericID = businessNumID[document["business_id"].GetString()];

		user = &allUsers[userNumericID];
		busines = &allBusiness[businessNumericID];

		user->businessReviewed.push_back(businessNumericID);
		user->stars[businessNumericID] = document["stars"].GetDouble();

		busines->usersReviewed.push_back(userNumericID);
		busines->stars[userNumericID] = document["stars"].GetDouble();
	}
}

