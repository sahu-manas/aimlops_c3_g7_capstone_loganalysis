# aimlops_c3_g7_capstone_loganalysis
IISc TalentSprint AI ML Ops Cohort 3 Group 7 Capstone Project - Anomaly detection and RCA from log file analysis using LogBERT and DeepLog 

1. Clone the repository
2. Create a Python environment (python -m venv .env) and activate
3. cd aimlops_c3_g7_capstone_loganalysis/openstack_loganalysis
4. pip install -r requirements.txt -> this will install all the dependencies
5. You will need to create an openstack log file "openstack.log" and place that in the directory openstack_loganalysis/data
   i. Open a text editor and merge the following files
   - deeplog/example/data/OpenStack/openstack_normal1.log
   - deeplog/example/data/OpenStack/openstack_normal2.log
   - deeplog/example/data/OpenStack/openstack_abnormal.log
   Merge the content of these three files into one file and name it as openstack.log and place that in the directory openstack_loganalysis/data

6. cd openstack_loganalysis
7. python data_process.py (this will create a folder called output and create the processed data sets)
8. python logbert.py vocab
9. python logbert.py train
10. python logbert.py predict
