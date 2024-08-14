import pandas as pd
import os
import numpy as np


COLUMNS_COVER = ['Parameter_names', 'Parameter_default_values']


class SummaryLogger():
    def __init__(self, all_args, all_default_args, out_path, approach):
        self.out_path = out_path
        self.approach = approach

        self.all_default_args = all_default_args
        self.list_parameter_names = list(all_args.keys())
        self.list_parameter_values = list(all_args.values())

        
        self.incdec_columns = ['Model_name'] + self.list_parameter_names + ['mAP', 
                                                                     'weighted_mAP',
                                                                     ]
        self.parameters_columns = ['Model_name'] + self.list_parameter_names
      

   



    def update_summary(self, exp_name, logger, avg_time):
        list_map = list(np.around(logger.mean_ap, decimals=3))
        list_map_weighted = list(np.around(logger.map_weighted, decimals=3))
        


        df = pd.DataFrame([[exp_name]+ 
                            self.list_parameter_values+ 
                            ["#".join(str(item) for item in list_map)]+
                            ["#".join(str(item) for item in list_map_weighted)]], columns=self.incdec_columns)
        
        df_time = pd.DataFrame([avg_time],columns=['Avg_Time_per_ep'])
        df_time.to_csv(os.path.join(self.out_path, exp_name,  "avg_time.csv"), index=False)
        
        df.to_csv(os.path.join(self.out_path, exp_name,  "summary.csv"), index=False)
    

   

    def summary_parameters(self, exp_name):

        df = pd.DataFrame([[exp_name]+ 
                                self.list_parameter_values], columns=self.parameters_columns)      
        df.to_csv(os.path.join(self.out_path, exp_name,  "parameters.csv"), index=False)
    

   

 