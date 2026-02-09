# gymnasium
https://gymnasium.farama.org/
## github
https://github.com/Farama-Foundation/Gymnasium
## MarkdownViewer++ toggle
    ctrl-shft-M
## python setup
### install Python install manager from Microsoft Store app.
https://apps.microsoft.com/detail/9nq7512cxl7t
### check highest required python version
https://github.com/Farama-Foundation/Gymnasium
#### install lower version if neccessary
    py install 3.13
#### add python executables to path of *current* user
    %USERPROFILE%\AppData\Local\Python\bin
### Create virtual environment
#### Git bash
    python3.13.exe -m venv venv/3.13
### activate virtual environment
#### Git bash
    source venv/3.13/bin/activate
### upgrade pip
    python -m pip install --upgrade pip
### install gymnasium
    pip install gymnasium[classic-control]
## xonsh
### .config\xonsh\rc.xsh

	aliases['ll'] = 'ls -la --color=auto'

	# import python modules from a local directory
	# https://xon.sh/customization.html#import-python-modules-from-a-local-directory

	import sys
	sys.path.insert(0, '')	
	   