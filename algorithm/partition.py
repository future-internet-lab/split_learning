import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Add topo")
parser.add_argument('--topo', type=int, nargs='+', help="Topo", required=False, default=[0])
args = parser.parse_args()
topo = args.topo

t_exe_1 = [13357545.69, 16633400.59, 4483398.08, 108550489.29, 17601796.66, 6936498.06,
           25696338.27, 47776377.32, 8637623.77, 2235370.44, 96616254.5, 8986239.39, 4386030.65,
           14442040.75, 42552181.87, 5193747.46, 1090723.94, 86274202.48, 4755385.98, 2229564.38,
           86263687.08, 4272859.6, 2312920.07, 8134938.78, 36749123.82, 2874437.36, 519684.34,
           73554284.92, 2281381.1, 503795.99, 73459154.57, 2450731.55, 531895.94, 4223146.0,
           21763522.16, 1357753.55, 156551.17, 20444858.76, 1271924.37, 147437.56, 20153109.71,
           1279938.39, 150243.76, 1442851.39, 28792.79, 259277.77, 6840649.43, 284488.51, 918151.45,
           68497649.07, 312602.02, 615632.38]
t_exe_2 = [9626523.11, 9744432.79, 4286511.46, 60510224.05, 10576457.07, 5097205.73, 12906717.74,
           26376752.77, 4992485.95, 1810399.8, 50717938.73, 5193841.16, 2146590.37, 7421044.63, 23193000.18, 3154655.87,
           848866.48, 45174193.11, 2575126.57, 1060644.18, 46332401.22, 2693396.78, 1096505.41, 4220884.39, 19785846.25,
           1460282.29, 315123.36, 39066106.12, 1223387.68, 326484.06, 38854006.72, 1235939.12, 332266.52, 2233031.84,
           11559281.41, 903486.26, 95082.83, 11157435.01, 911345.66, 92444.06, 10991967.75, 922106.88, 92057.63,
           806132.14, 25883.7, 249798.98, 3717061.88, 201992.79, 619494.72, 34568209.2, 223099.61, 422914.34]
t_exe_3 = [1842203.11, 134938.37, 125033.87, 1686346.34, 256914.02, 19534.08, 120268.93,
           432535.91, 2079241.13, 21802.23, 823672.03, 557308.75, 339996.65, 534434.17,
           5355870.75, 884655.06, 443887.27, 1163461.4, 275841.08, 168527.39, 1648078.27,
           275590.77, 168415.56, 223443.76, 1092814.22, 264937.37, 67833.73, 1357821.23,
           265718.0, 67843.01, 1344730.26, 207879.32, 153451.22, 135037.95, 941600.35,
           346008.57, 272910.41, 1394146.17, 346735.06, 272310.82, 1393472.19, 111306.09, 290160.6,
           271277.18, 8299.43, 1164661.09, 242398.71, 19644.53, 21380.09, 44370.26, 17210.62, 594003.99]
size_data = [33554432, 33554432, 33554432, 33554432, 33554432, 33554432, 8388608, 16777216, 16777216, 16777216,
             16777216, 16777216, 16777216, 4194304, 8388608, 8388608, 8388608, 8388608, 8388608, 8388608, 8388608,
             8388608, 8388608, 2097152, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304,
             4194304, 1048576, 1048576, 1048576, 1048576, 1048576, 1048576, 1048576, 1048576, 1048576, 1048576,
             262144, 262144, 262144, 2097152, 2097152, 2097152, 2097152, 2097152, 5120]

# LAN
a1_2 = 5.321956086901455
a2_3 = 11.684686209372796
# 5G
# a1_2 = 50
# a2_3 = 50

# layer1_exe = t_exe_1
# layer1_comm_data = [x * a1_2 for x in size_data]
# layer2_exe = t_exe_1
# layer2_comm_data = [x * a2_3 for x in size_data]
# layer2_comm_grad = [x * a1_2 for x in size_data]
# layer3_exe = t_exe_3
# layer3_comm_grad = [x * a2_3 for x in size_data]

layer1_exe = t_exe_1
layer1_comm_data = [x * a1_2 for x in size_data]
layer2_exe = t_exe_1
layer2_comm_data = [x * a1_2 for x in size_data]
layer2_comm_grad = [x * a1_2 for x in size_data]
layer3_exe = t_exe_1
layer3_comm_grad = [x * a1_2 for x in size_data]
time_min = float('inf')
result = None

training_time_rate = 3

if len(topo) == 2:
    for i in range(1, len(size_data) - 1):
        layer1 = (sum(layer1_exe[:i]) * (training_time_rate + 1) + layer1_comm_data[i - 1]) / topo[0]
        layer2 = (sum(layer2_exe[i:]) * (training_time_rate + 1) + layer2_comm_grad[i - 1]) / topo[1]

        time_max = max(layer1, layer2)
        if time_max < time_min:
            result = [i]
            time_min = time_max
if len(topo) == 3:
    for i in range(1, len(size_data) - 1):
        for j in range(i + 1, len(size_data)):
            layer1 = (sum(layer1_exe[:i]) * (training_time_rate + 1) + layer1_comm_data[i - 1]) / topo[0]
            layer2 = (sum(layer2_exe[i:j]) * (training_time_rate + 1) + layer2_comm_data[j - 1] + layer2_comm_grad[i - 1]) / topo[1]
            layer3 = (sum(layer3_exe[j:]) * (training_time_rate + 1) + layer3_comm_grad[j - 1]) / topo[2]

            time_max = max(layer1, layer2, layer3)
            if time_max < time_min:
                result = [i, j]
                time_min = time_max

print(f"Partition at: {result} - {np.array(size_data)[result]}")
print(f"Time min = {time_min/1000000000} s")

# print(sum(t_exe_1) * (training_time_rate + 1) / 1000000000)
# print(sum(t_exe_2) * (training_time_rate + 1) / 1000000000)
# print(sum(t_exe_3) * (training_time_rate + 1) / 1000000000)
