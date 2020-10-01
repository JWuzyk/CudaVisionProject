import os
import pandas as pd

class Logger():
    def __init__(self,runs_path,log_path,run_id,args):
        self.log_path = log_path
        self.runs_path = runs_path
        self.run_id= run_id
        self.args = args
        self.args['run_id']=run_id
        del self.args['size_after_transform']
        run = pd.DataFrame(self.args,index = [run_id])
        
        if os.path.isfile(self.runs_path):
            run.to_csv(runs_path, mode='a', header=False)
        else:
            run.to_csv(runs_path, mode='a', header=True)

        self.log = pd.DataFrame(columns=['model_id','Epoch', 'Train Loss det', 'Train Loss seg', 'Test Loss det', 'Test Loss seg', 
                                    'Balls correct' ,'Balls wrong' ,'Balls FP',
                                    'Goals correct' ,'Goals wrong' ,'Goals FP',
                                    'Robots correct' ,'Robots wrong' ,'Robots FP',
                                    'Background IOU','Lines IOU','Field IOU',
                                    'Background Acc','Lines Acc','Field Acc','time'])
        self.log.loc[0,'Epoch']=0
        
    def log_data(self):
        if os.path.isfile(self.log_path):
            self.log.to_csv(self.log_path, mode='a', header=False)
            print('writing logs to ', self.log_path)
        else:
            print('making log csv')
            self.log.to_csv(self.log_path, mode='a', header=True)
            print('writing logs to ', self.log_path)
        self.log.loc[0,'Epoch']+=1

