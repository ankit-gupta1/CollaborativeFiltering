/*
 * utils.cpp
 *
 *  Created on: Feb 16, 2015
 *      Author: Ankit
 */

#include <iostream>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <sstream>
#include "utils.h"
#include "collab_filtering.h"

string getCurrentTimeString() {
	time_t rawtime;
	struct tm * timeinfo;
	char buffer[80];

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	strftime(buffer, 80, "%d_%m_%Y_%I_%M_%S", timeinfo);
	string str(buffer);

	return str;
}

void logUserBusinessFeatures(double **u, double **v, unsigned int totalUsers,
		unsigned int totalBusiness) {
	ofstream fout;
	string userFileName;
	string businessFileName;
	ostringstream sl;
	ostringstream si;

	sl << LATENT_SPACE;
	si << MAX_ITERATION;

	/* Save the user features. */
	userFileName = "..\\..\\dump\\U_";
	userFileName.append(getCurrentTimeString());
	userFileName.append("_LS_");
	userFileName.append(sl.str());
	userFileName.append("_ITR_");
	userFileName.append(si.str());
	userFileName.append(OP_FILENAME_EXT);
	char ufn[userFileName.length() + 1];
	copy(userFileName.begin(), userFileName.end(), ufn);
	ufn[userFileName.length()] = '\0';
	fout.open(ufn);

	for (unsigned int i = 0; i < totalUsers; i++) {
		fout << "u " << setw(6) << i << ", ";
		for (unsigned int j = 0; j < LATENT_SPACE; j++) {
			fout << setw(13) << setprecision(5) << u[i][j] << ", ";
		}

		fout << endl;
	}

	fout.close();

	/* Save the business features. */
	businessFileName = "..\\..\\dump\\V_";
	businessFileName.append(getCurrentTimeString());
	businessFileName.append("_LS_");
	businessFileName.append(sl.str());
	businessFileName.append("_ITR_");
	businessFileName.append(si.str());
	businessFileName.append(OP_FILENAME_EXT);
	char ufn1[businessFileName.length() + 1];
	copy(businessFileName.begin(), businessFileName.end(), ufn1);
	ufn1[businessFileName.length()] = '\0';
	fout.open(ufn1);

	for (unsigned int i = 0; i < totalBusiness; i++) {
		fout << "v " << setw(6) << i << ", ";
		for (unsigned int j = 0; j < LATENT_SPACE; j++) {
			fout << setw(13) << setprecision(5) << v[i][j] << ", ";
		}

		fout << endl;
	}

	fout.close();
}
