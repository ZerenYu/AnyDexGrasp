import os
def get_date(in_path, write_path):
    with open(write_path, 'w') as f:
        for date in sorted(os.listdir(in_path)):
            f.write(date + '\n')
def get_dh3_date(in_path, write_path):
    dates = []
    for pose in os.listdir(in_path):
        for date in os.listdir(os.path.join(in_path, pose)):
            dates.append(date)
    with open(write_path, 'w') as f:
        for date in sorted(dates):
            f.write(date + '\n')
in_path = '/home/ubuntu/data/hengxu/data/inspire/inspire_test_single_point/random'
write_path = '/home/ubuntu/data/hengxu/data/inspire/inspire_test_single_point/inspire_random.txt'
get_date(in_path, write_path)    
