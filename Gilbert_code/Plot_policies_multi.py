import matplotlib.pyplot as plt

policy_results = {
        "sumo\nL": 90.2,
        "sumo\nH": 134.59,
        "43_L": 177.97,
        "43_H": 273.2,
        "80_L": 179.58,
        "80_H": 290.23,
        "160_L": 186.6,
        "160_H": 211.5,
        "240_L": 188.7,
        "240_H": 196.01
}

plt.bar(list(policy_results.keys())[0:2], list(policy_results.values())[0:2], color="red")
plt.bar(list(policy_results.keys())[2:], list(policy_results.values())[2:])

# display hist and save hist
plt.ylabel("Average travel time (sec)")
plt.title("Policy vs Actuated Travel times:\n "
          "1x3 Grid")
plt.savefig("policy_1x3.png")
# plt.show()

plt.clf()

policy_results = {
        "sumo\nL": 88.55,
        "sumo\nH": 152.26,
        "43_L": 165.43,
        "43_H": 241.5,
        "80_L": 165.4,
        "80_H": 243.04,
        "160_L": 267.5,
        "160_H": 240.95,
        "240_L": 174.2,
        "240_H": 241.9
}

plt.bar(list(policy_results.keys())[0:2], list(policy_results.values())[0:2], color="red")
plt.bar(list(policy_results.keys())[2:], list(policy_results.values())[2:])

# display hist and save hist
plt.ylabel("Average travel time (sec)")
plt.title("Policy vs Actuated Travel times:\n "
          "2x2 Grid")
plt.savefig("policy_2x2.png")
# plt.show()
