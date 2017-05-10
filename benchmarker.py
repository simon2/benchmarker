import importlib
import json
import os
import datetime
import begin
import sys
import pkgutil
from sysinfo import sysinfo

sys.path.append("modules")
sys.path.append("data_helpers")


def get_time_str():
    d = datetime.datetime.now()
    s = d.strftime("%y.%m.%d_%H.%M.%S")
    return s


def gen_name_output_file(params):
    name = "{}_{}_{}_{}.json".format(
        params["problem"],
        params["framework"],
        params["device"],
        get_time_str()
        )
    return name


def save_params(params):
    str_result=json.dumps(params, sort_keys=True,  indent=4, separators=(',', ': '))
    print(str_result)
    path_out = params["path_out"]
    name_file = gen_name_output_file(params)
    with open(os.path.join(path_out, name_file), "w") as f:
        f.write(str_result)


def get_modules():
    path_modules="modules"
    return [name for _, name, is_pkg in pkgutil.iter_modules([path_modules]) if not is_pkg and name.startswith('do_')]


@begin.start
def main(framework: "Framework to test" = "numpy",
         problem: "problem to solve" = "2048",
         path_out: "path to store results" = "./logs",
         gpus: "list of gpus to use" = ""
         ):
    if len(sys.argv) < 2:
        print("available frameworks :")
        for plugin in get_modules():
            print("\t", plugin[3:])
        return

    params = sysinfo.get_sys_info()
    params["framework"] = framework
    params["path_out"] = path_out
    if len(gpus)>0:
        params["gpus"] = list(map(int, gpus.split(',')))
    else:
        params["gpus"] = []

    params["nb_gpus"] = len(gpus)
    params["problem"] = problem

    if params["nb_gpus"] > 0:
        params["device"] = params["gpu"]
    else:
        params["device"] = params["cpu_brand"]

    mod = importlib.import_module("modules.do_"+params["framework"])
    run = getattr(mod, 'run')

    params = run(params)
    save_params(params)
