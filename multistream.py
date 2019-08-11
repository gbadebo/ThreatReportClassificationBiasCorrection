import sys
from manager import Manager
from properties import Properties

"""
Parameters
datasetName: main part of dataset file name, e.g., powersupply for powersupply_source_stream.csv, powersupply_target_stream.csv
baseline: 1=startMscKLIEP, 2=start, 3=start2, 4=start_skmm, 5=start_mkmm, 6=start_srconly, 7=start_trgonly
"""
def main(datasetName, method):
	if method not in ['kmm','kliep','arulsif']:
		print('Methods allowed are : kmm, kliep or arulsif. Please try again.')
		return

	props = Properties('config.properties', datasetName)
	srcfile = Properties.BASEDIR + datasetName + Properties.SRCAPPEND
	trgfile = Properties.BASEDIR + datasetName + Properties.TRGAPPEND
	mgr = Manager(srcfile, trgfile)

	Properties.logger.info(props.summary(method))
	Properties.logger.info('Start classification for biased label data.')

	mgr.startClassification(datasetName, method)
"""
if __name__ == '__main__':
	main()
"""

