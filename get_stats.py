import sys, json, numpy

#Get read the file and get the info

file_location = sys.argv[1]

with open(file_location, 'r') as f:
    hello = json.load(f)

results = [hi[1] for hi in hello]
print('Best performance:',results[0])
print('Worst performance: ', results[len(results)-1])
print('Mean performance: ', numpy.mean(results))
print('Std dev: ', numpy.std(results))
print('Std error: ', numpy.std(results)/numpy.sqrt(len(results)))
