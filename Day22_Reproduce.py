from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/"water_potability_temporal_synthetic.csv")
df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
sensor_cols=[c for c in df.columns if c not in ["timestamp","Potability"]]
data=df.dropna(subset=["timestamp"]).sort_values("timestamp",kind="stable").reset_index(drop=True)
data["target_next_ph"]=data["ph"].shift(-1)
data=data.dropna(subset=["target_next_ph"]).reset_index(drop=True)
n=len(data); a=int(n*.70); b=int(n*.85)
train,val,test=data.iloc[:a],data.iloc[a:b],data.iloc[b:]
def linear():
 return Pipeline([("imputer",SimpleImputer(strategy="median")),("model",LinearRegression())])
def forest(cols):
 return Pipeline([("imputer",SimpleImputer(strategy="median")),
  ("model",RandomForestRegressor(n_estimators=150,max_depth=12,min_samples_leaf=4,max_features=.8,random_state=42,n_jobs=-1))])
candidates=[("linear_regression_pH_only",["ph"],linear()),
 ("random_forest_pH_only",["ph"],forest(["ph"])),
 ("random_forest_full_9_features",sensor_cols,forest(sensor_cols)),
 ("mean_baseline",["ph"],Pipeline([("imputer",SimpleImputer(strategy="median")),("model",DummyRegressor(strategy="mean"))]))]
rows=[]
for name,cols,m in candidates:
 m.fit(train[cols],train.target_next_ph); p=m.predict(val[cols])
 rows.append({"candidate":name,"MAE":mean_absolute_error(val.target_next_ph,p),
              "RMSE":np.sqrt(mean_squared_error(val.target_next_ph,p))})
p=val.ph.fillna(train.ph.median()).to_numpy()
rows.append({"candidate":"persistence_current_pH","MAE":mean_absolute_error(val.target_next_ph,p),
             "RMSE":np.sqrt(mean_squared_error(val.target_next_ph,p))})
print(pd.DataFrame(rows).sort_values("RMSE"))
final=linear(); dev=pd.concat([train,val],ignore_index=True)
final.fit(dev[["ph"]],dev.target_next_ph)
pred=final.predict(test[["ph"]])
print("Reserved test MAE:",mean_absolute_error(test.target_next_ph,pred))
print("Reserved test RMSE:",np.sqrt(mean_squared_error(test.target_next_ph,pred)))
joblib.dump(final,ROOT/"Day22_Final_Linear_Regression_Pipeline_reproduced.joblib")
