#!/usr/bin/env bash
# Initialize the development environment. It's best to source this file, i.e.
# source init_dev_env.sh, in order to have the virtualenv enironment active.

pyvenv pyvenv > /dev/null && source ./pyvenv/bin/activate
if [[ -z $VIRTUAL_ENV ]]
then
  echo "Could not create or activate the pyvenv environment."
  exit 1
fi

echo "Installing python's libraries"
pip install -r ./requirements.txt

ln -s ./pyvenv/bin/activate activate
