import os
import sys

class SumoUtil:
    def add_sumo_path():
        if "SUMO_HOME" in os.environ:
            tools = os.path.join(os.environ["SUMO_HOME"], "tools")
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

    def get_sumo_binary(use_gui=True):
        from sumolib import checkBinary
        if "SUMO_HOME" in os.environ:
            bin_path = os.path.join(os.environ["SUMO_HOME"], "bin")
            sys.path.append(bin_path)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        # using sumo or sumo-gui
        if use_gui:
            return checkBinary("sumo-gui")
        checkBinary("sumo")
