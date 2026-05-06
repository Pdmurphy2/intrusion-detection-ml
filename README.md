# intrusion-detection-ml
Python Machine Learning final repo

## Our Running App
You can take a look at our app and how it functions in the running_app folder and see screenshots of it running. 

    - First is the login screen where it starts the timer and tracks your failed and total logins
    - After you login with username: admin and password: password123 you can enter in values for our features
        - ip_reputation values closer to 0 are considered normal and closer to 1 are considered attacks.
        - network_packet_size in between 70 and 7000 are considered normal behavior and outside of that are suspicious.
        - our model also relies heavily on faild_logins and login_attempts the higher they are the more likely to be attacks.
        - unusual_time_access value 1 is true so it is not normal business hours and 0 is false.
    - The last screen is the summary showing the values you entered and whether our model predicts what you entered as an attack or not.


