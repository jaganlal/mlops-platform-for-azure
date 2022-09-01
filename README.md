# mlops-platform-for-azure
Simple MLOps platform for Azure

# References
https://docs.microsoft.com/en-us/azure/machine-learning/v1/how-to-deploy-azure-container-instance

# Login from Mac

az login --tenant 95917aa5-d840-4443-a19e-aa3ad09d4bb5

az ml folder attach -w simple_mlops_demo_workspace -g RG_Jaganlal

az ml datastore show-default

az ml service show -n mlops-deploy-service