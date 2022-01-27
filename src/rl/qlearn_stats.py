import os
import sys
import json

import rospkg

# Get ros package path
ros_ = rospkg.RosPack()
ros_path = ros_.get_path('deep-rl-pusher')

def loadConfig(config):
    data = None
    with open(config) as file:
        data = json.load(file)
    return data

cfg = loadConfig(os.path.join(ros_path + "/config/actions.config"))

def loadQTable(file):
    import os
    import pickle
    folder = os.path.join(ros_path, 'tmp')
    file_p = os.path.join(folder, file)
    file = open(file_p, "rb")
    q_table = pickle.load(file)
    return q_table

def qlearnStats(table):
    print(len(table))
    print("{:<8} {:<15} {:<10}".format('State','Action','QValue'))
    for k, v in table.items():
        (coords, a) = k
        print("{:<8} {:<15} {:<10}".format(coords, a, v))
    print(len(table))

if __name__ == '__main__':
    print('<QLearn Stats>')
    table = loadQTable(sys.argv[1])
    qlearnStats(table)
    print('<Done>')