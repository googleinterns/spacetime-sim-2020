# import matplotlib.pyplot as plt
#
# policy_results = {
#         "sumo\nL": 56.6,
#         "sumo\nH": 97.74,
#         "43_L": 73.35,
#         "43_H": 109.98,
#         "80_L": 61.58,
#         "80_H": 147.2,
#         "160_L": 83.13,
#         "160_H": 123.8,
#         "240_L": 85.50,
#         "240_H": 173.37
# }
#
# plt.bar(list(policy_results.keys())[0:2], list(policy_results.values())[0:2], color="red")
# plt.bar(list(policy_results.keys())[2:], list(policy_results.values())[2:])
#
# # display hist and save hist
# plt.ylabel("Average travel time (sec)")
# plt.title("Policy vs Actuated Travel times:\n "
#           "1x1 Grid")
# plt.savefig("policy_1x1.png")
# plt.show()
#

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(15, 8), facecolor='w', edgecolor='k')

policy_results = {
        "sumo\nL": 56.6,
        "sumo\nH": 97.74,
        "5x5\nL": 55.82,
        "5x5\nH": 132.0,
        "10x10\nL": 55.23,
        "10x10\nH": 107.5,
        "5x5x5\nL": 175.58,
        "5x5x5\nH": 165.6,
        "10x10x10\nL": 53.4,
        "10x10x10\nH": 107.76,
        "32x32x32\nL": 55.83,
        "32x32x32\nH": 145.06
}

plt.bar(list(policy_results.keys())[0:2], list(policy_results.values())[0:2], color="red")
plt.bar(list(policy_results.keys())[2:], list(policy_results.values())[2:])


# plt.bar(list(policy_results.keys())[::2], list(policy_results.values())[::2])
# plt.bar(list(policy_results.keys())[0], list(policy_results.values())[0], color="red")

# plt.bar(list(policy_results.keys())[1::2], list(policy_results.values())[1::2])
# plt.bar(list(policy_results.keys())[1], list(policy_results.values())[1], color="red")

# display hist and save hist
plt.ylabel("Average travel time (sec)")
plt.title("Policy vs Actuated Travel times:\n "
          "1x1 Grid")
# plt.savefig("testing_policy_1x1.png")
# plt.show()


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(15, 8), facecolor='w', edgecolor='k')

policy_results = {
        "sumo\nL": 56.6,
        "sumo\nH": 97.74,
        "5x5\nL": 55.82,
        "5x5\nH": 132.0,
        "10x10\nL": 55.23,
        "10x10\nH": 107.5,
        "5x5x5\nL": 175.58,
        "5x5x5\nH": 165.6,
        "10x10x10\nL": 53.4,
        "10x10x10\nH": 107.76,
        "32x32x32\nL": 55.83,
        "32x32x32\nH": 145.06
}

plt.bar(list(policy_results.keys())[0:2], list(policy_results.values())[0:2], color="red")
plt.bar(list(policy_results.keys())[2:], list(policy_results.values())[2:])


# plt.bar(list(policy_results.keys())[::2], list(policy_results.values())[::2])
# plt.bar(list(policy_results.keys())[0], list(policy_results.values())[0], color="red")

# plt.bar(list(policy_results.keys())[1::2], list(policy_results.values())[1::2])
# plt.bar(list(policy_results.keys())[1], list(policy_results.values())[1], color="red")

# display hist and save hist
plt.ylabel("Average travel time (sec)")
plt.title("Policy vs Actuated Travel times:\n "
          "1x1 Grid")
# plt.savefig("testing_policy_1x1.png")
# plt.show()





