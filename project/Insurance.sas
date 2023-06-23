/* Dataset source : https://reurl.cc/KM710g */
/* STEP: EDA(Exploratory Data Analysis) -> Linear Regression */
/* EDA */
PROC IMPORT
		DATAFILE="C:\Users\zhuol\Desktop\sas imfo\program\archive\insurance.csv" DBMS=CSV
		OUT=insu;
		GETNAMES=YES;
RUN;
/* Import dataset form my desktop*/

PROC PRINT DATA=insu;
RUN;
PROC CONTENTS DATA=insu;
RUN;
PROC MEANS DATA=insu N NMISS;
TITLE "Missing value check for the data set";
RUN;
/* Do a quick look of the data */
/*The sum of observations for each factor is 1338, so there are no missing values in the raw data */


PROC SORT DATA=insu nouniquekey OUT= rep;
		BY age bmi charges;
PROC PRINT DATA= rep;
RUN;

PROC SORT DATA=insu nodupkey OUT=insu2;
		BY age bmi charges;
RUN;
/* Find a duplicate value (age=19,bmi=1639.5631)by using 'nouniquekey' and delete it. */


PROC SGPLOT DATA=insu2;
		HISTOGRAM charges;
DENSITY charges/ type=kernel;
TITLE "Histogram for charges";
RUN;
TITLE;
PROC MEANS DATA=insu2 SKEWNESS KURTOSIS;
		VAR charges;
RUN;
/* Skewness>0 --> Positive skew ; Kurtosis >0 --> Peak occurs ,
	so I tried to log charges to close normal distribution */


DATA cleaninsu;
		SET insu2;
		log_charges = log(charges);

		IF sex = "male" THEN sex= 1;
		ELSE sex= 0;
		n_sex = input(sex, comma9.);
		DROP sex;

		IF smoker = "no" THEN smoker=0;
		ELSE smoker=1;
		n_smoker = input(smoker, comma9.);
		DROP smoker;

		IF region = "northeast" THEN region=1;
		ELSE IF region="northwest" THEN region=2;
		ELSE IF region="southeast" THEN region=3;
		ELSE IF region="southwest" THEN region=4;
		n_region  = input(region, comma9.);
		DROP region;
RUN;
/* Convert categorical data into numerical codes.*/

PROC CONTENTS DATA=cleaninsu;
RUN;
/* Check whether the dataset turned out the way I wanted */;


PROC CORR DATA=cleaninsu ;
		VAR age n_sex bmi children n_smoker n_region log_charges;
RUN; QUIT;
/* Create a correlation matrix, and find smoker have best relation with charges*/

PROC SGPLOT DATA=cleaninsu;
		HBOX log_charges / CATEGORY = n_smoker;
PROC SGPLOT DATA=cleaninsu;
		HBOX log_charges / CATEGORY = n_sex;
PROC SGPLOT DATA=cleaninsu;
		HBOX log_charges / CATEGORY = n_region;
PROC SGPLOT DATA=cleaninsu;
		HBOX log_charges / CATEGORY = children;
RUN;


PROC SGSCATTER DATA=cleaninsu;
		COMPARE y=log_charges
		x=(age bmi n_region) / GROUP=n_smoker;
PROC SGSCATTER DATA=cleaninsu;
		COMPARE y=log_charges
		x=(age bmi n_smoker) / GROUP=n_sex;
RUN;
/* Based on the matrix, scatter plot and box plot, I believe that smoking and charges have the strongest correlation. */

PROC REG DATA=cleaninsu OUTEST=Regout;
		M1: MODEL log_charges = age n_sex children n_smoker n_region bmi / P;
		OUTPUT OUT=Predicted PREDICTED=pred_charges;
RUN; QUIT;
/* Create a regression and save predicted values to compare with actual data. */


PROC SGPLOT DATA=Predicted;
		SCATTER x=pred_charges y=log_charges/ MARKERATTRS = (COLOR = BLUE SYMBOL = CIRCLE) NAME="PREDICTED";
		REG x=pred_charges y=log_charges / CLM;
RUN;
/* Compare predicted values with actual data */
