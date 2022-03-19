# CPS824 Problem Set Instructions

## Setup for Coding Parts

1. You need to setup the needed python packages using `pip install -r requirements.txt`, with the requirements.txt file that 
   comes in the assignment folder.
    
2. For Editor I suggest you use [PyCharm](https://www.jetbrains.com/pycharm/). This is just a suggestion and feel free
to use the editor of your choice.
   
3. We will be using [Open AI gym](https://gym.openai.com/docs/) for the coding portion of this assignment. 
   Feel free to look at its webpage for more details. The requirements.txt should automatically install gym for you.
   In this assignment we will be using [Frozen Lake environment](https://gym.openai.com/envs/FrozenLake-v0).


## (Optional) Setup for Coding Parts Using Virtual Environment
Once you have unzipped the starter code, you might want to create a 
[virtual environment](https://docs.python-guide.org/dev/virtualenvs/) 
for the project. If you choose not to use a virtual environment, it is up to you to make sure that 
all dependencies for the code are installed on your machine. To set up a virtual environment, run the following:
1. Change the directory to the "src" directory in the assignment folder.
2. Run `sudo pip install virtualenv` to install virtual environment. Note that this may be already installed. 
   Note that depending on your system you may need use `pip3` instead of `pip`.
3. Run `virtualenv .env` to create the virtual environment.
4. Run `source .env/bin/activate` to activate the virtual environment.
5. Run `pip install -r requirements.txt` to install the dependencies.
6. If you are finished with the assignment and would like to deactivate the virtual environment, you can run 
   `deactivate`.
