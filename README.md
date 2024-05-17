# CSE 587 Final Project

Set up conda environment :

`conda create -n dic587semproj python=3.11`

Remember to always activate your environment before starting any coding. Add some shortcut at the end of your `.bashrc` or `.bash_profile` or `.zshrc` file if you want.

`alias dic="conda activate dic587semproj"`

Don't forget to run this command to have this effect immediately in your current shell :

`source ~/.zshrc` [or whatever file you updated with `alias`]

If you're installing a new package in this environment, update the `requirements.txt` with this :

`pip list --format=freeze > requirements.txt`