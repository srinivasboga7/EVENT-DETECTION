import csv
import sort_timestamps as srt

emails_order, time = srt.main()
print(len(emails_order))

path_unsorted = "/home/sai/Documents/DATA MINING/PROJECT_EMAIL/DATA/test_email_data.csv"
path = "sorted_email_data.csv"
with open(path_unsorted,"r") as file :
		reader = csv.reader(file, delimiter = "\n")
		lines = list(reader)
		for i in emails_order:
			line = lines[i]
			# print(line)
			with open(path, "a") as fp:
				wr = csv.writer(fp, dialect='excel')
				wr.writerow(line)
