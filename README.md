Project to predict Dublin Bus arrival times.

Hi! We hope you enjoy using our Hustle Bustle app. Please find instructions for operating the app below.

Setup a virtual environment, HustleBustle Activate virtual environment • source activate HustleBustle Linux/Mac OS • activate HustleBustle Windows Install required packages by running python setup.py install

<h2>Flask Application</h2>

Below are the steps for operating the site on a local machine. The application is also currently live at www.HustleBustle.club.

Run the main.py file Go to localhost:5000 on your browser

To find a journey option: • Enter source and destination addresses • Choose preferred time to leave • Click on “Go”

To utilise “Get me Home” & “Get me to Work” • Register an account through the link on the right of the navbar • Enter a home/work address • Enable location sharing • Click on “Get me Home” / “Get me to Work”

<h2>Datahandling</h2>

Ensure raw data is in a folder called 'DayRawCopy' and this contains the AVL data in csv format, unzipped as supplied. This cannot be stored on github, please download this seperately.

To run datahandling.py • activate virtual environment • run code with python >python datahandling.py
