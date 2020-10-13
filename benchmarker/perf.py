import re
import subprocess
from benchmarker.util import abstractprocess

perf_counters_multipliers = {'r5302c7': 1,
                             'r5308c7': 4,
                             'r5320c7': 8}

def get_counters(output_dict, command):
    output_dict["problem"]["flop_measured"] = 0
    for counter in perf_counters_multipliers:
        perf_command = ["perf", "stat", "-e", counter]
        proc = abstractprocess.Process("local", command=perf_command + command)
        process_err = proc.get_output()["err"]
        #print(process_err)
        if process_err:
            match_exp = re.compile('[\d|\,]+\s+' + counter).search(process_err)
            if match_exp:
                match_list = match_exp.group().split()
                cntr_value = int(match_list[0].replace(',',''))
                output_dict["problem"]["flop_measured"] += perf_counters_multipliers[counter] * cntr_value
    return output_dict
