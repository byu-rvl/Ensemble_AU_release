# import sys
# import os
# import re
# import subprocess

# from PyQt5 import QtCore

# def isBuild():
#     """
#     Determine whether the app is frozen using pyinstaller/py2app.
#     Returns True when this is a release or nightly build (eg. it is build as a
#     distributable package), returns False if it is a source checkout.
#     """
#     return getattr(sys, 'frozen', False)

# def getCwd():
#     """
#     Retrieve the folder where makehuman.py or makehuman.exe is located.
#     This is not necessarily the CWD (current working directory), but it is what
#     the CWD should be.
#     """
#     if isBuild():
#         return os.path.dirname(sys.executable)
#     else:
#         return os.path.dirname(os.path.realpath(__file__))

# def set_sys_path():
#     """
#     Append local module folders to python search path.
#     """
#     #[BAL 07/11/2013] make sure we're in the right directory
#     if not sys.platform.startswith('darwin'): # Causes issues with py2app builds on MAC
#         print("in if________________________________________________________________________________", getCwd())
#         os.chdir(getCwd())
#         syspath = ["./", "./lib", "./apps", "./shared", "./apps/gui","./core"]
#         syspath.extend(sys.path)
#         sys.path = syspath
#     else:
#         print("IN ELSE HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1------------------------------------------")
#         data_path = "../Resources/makehuman"
#         if(os.path.isdir(data_path)):
#             os.chdir(data_path)
#         syspath = ["./lib", "./apps", "./shared", "./apps/gui", "./core", "../lib", "../"]
#         syspath.extend(sys.path)
#         sys.path = syspath

#     if isBuild():
#         # Make sure we load packaged DLLs instead of those present on the system
#         os.environ["PATH"] = '.' + os.path.pathsep + getCwd() + os.path.pathsep + os.environ["PATH"]


# if __name__ == "__main__":
#     set_sys_path()

#     from mhmain import MHApplication
#     import human
#     import files3d
#     import mh
#     application = MHApplication()
#     # application.run()
#     # print("after running")
#     # application.loadHuman()
#     # print("after load human")
#     # import core
#     from core import G # can be omitted in MakeHuman's internal shell
#     G.app.selectedHuman = G.app.addObject(human.Human(files3d.loadMesh(mh.getSysDataPath("3dobjs/base.obj"), maxFaces = 5))) #init the human
#     human = G.app.selectedHuman

#     dir(human)

#     print(human.targetsDetailStack)
#     print(human)

#     # someTarget is a target name of your choice, weight is a float with 0.0 <= weight <= 1.0. A target is the equivalent to Blender's 'shape key' and weight is the amount this shape key will be applied. 

#     import getpath as gp

#     someTarget = "full_target"
#     weight = 0.5
#     human.setDetail(gp.getSysDataPath('targets/' + someTarget + '.target'),
#                     weight)  # setDetail takes the path to a target file and a weight (or strength) value (actually you should use os.path.join.... 8-B)

#     human.applyAllTargets()

#     age = human.getAgeYears()
#     print("age", age)
#     human.setAgeYears(20)
#     age = human.getAgeYears()
#     print("age after", age)

#     # close_standard_streams()
#     print("ends here")
