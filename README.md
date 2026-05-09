# intrusion-detection-ml
Python Machine Learning final repo

## Our Model
- We set out to make a machine learning model that given some input can tell whether the user behavior is normal or malicious.
- Our model was trained on this intrusion detection dataset from kaggle: https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset/data
    - We had some issues with the dataset in that some of the total logins were greater than the failed logins which we show in our notebook.

## Setup 
- You can see in our requrements.txt file which is what we used to make our notbook and our app.
- For the app we used flask and you can run it localy with the command: flask run. 

## Our Running App
You can take a look at our app and how it functions in the running_app folder and see screenshots of it running. 

- First is the login screen where it starts the timer and tracks your failed and total logins.
- After you login with username: admin and password: password123 you can enter in values that correlate with our features.
    - ip_reputation values closer to 0 are considered normal and closer to 1 are considered attacks.
    - network_packet_size in between 70 and 6000 are considered normal behavior and outside of that are suspicious.
    - our model also relies heavily on faild_logins and login_attempts the higher they are the more likely to be attacks.
    - unusual_time_access value 1 is true so it is not normal business hours and 0 is false meaning it is normal business hours.
- The last screen is the summary showing the values you entered and whether our model predicts what you entered as an attack or not.

## Locations
- Our filalized model is in the final_model folder.
- anything to do with the application is in the web_app folder.
    - screenshots of the running application is in the running_app folder inside of web_app.
- The judges evaluations and our self evaluations are inside of the evaluations folder. 
