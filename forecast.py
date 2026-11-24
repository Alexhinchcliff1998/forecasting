from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run():
    rng=np.random.default_rng(42); t=np.arange(260)
    df=pd.DataFrame({'week':t,'demand':200+0.2*t+35*np.sin(2*np.pi*t/52)+rng.normal(0,8,len(t))})
    df['lag1']=df.demand.shift(1); df['lag4']=df.demand.shift(4); df['season_sin']=np.sin(2*np.pi*t/52); df['season_cos']=np.cos(2*np.pi*t/52)
    df=df.dropna(); train,test=df.iloc[:-52],df.iloc[-52:]; features=['lag1','lag4','season_sin','season_cos']
    model=RandomForestRegressor(n_estimators=100,min_samples_leaf=3,random_state=42).fit(train[features],train.demand)
    pred=model.predict(test[features]); baseline=test.lag1
    metrics={'model_mae':float(mean_absolute_error(test.demand,pred)),'baseline_mae':float(mean_absolute_error(test.demand,baseline))}
    print(metrics); out=Path(__file__).parent
    pd.DataFrame({'week':test.week,'actual':test.demand,'prediction':pred,'baseline':baseline}).to_csv(out/'predictions.csv',index=False)
    plt.figure(figsize=(10,4));plt.plot(test.week,test.demand,label='Actual');plt.plot(test.week,pred,label='Model');plt.plot(test.week,baseline,label='Naive',alpha=.5);plt.xlabel('Week');plt.ylabel('Units');plt.legend();plt.tight_layout();plt.savefig(out/'forecast.png');plt.close()
    return metrics
if __name__=='__main__': run()
