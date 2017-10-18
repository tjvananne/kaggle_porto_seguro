# kaggle_porto_seguro
kaggle: who will create an auto insurance claim?

# 1) 999 level 1 small column sample xgb models

I'd like to take some checkpoints along the way with this strategy
When I hit 33 models in or so, this is the checkpoint I want to do:

* identify best `error`, `logloss`, and `auc` scoring single models
* load in trainA / trainB preds for those models and determine local score
* pass test through these models to determine the PLB scores (how well do they match local scores?)
* bag (mean) these top 3 model test scores together and pass into PBL (how does that compare to individual scores?)
* bag ALL 33 models so far and pass that to PBL, how does that score?


**Best Single Model Using AUC after 35 iters**

* level01 - 31
* features: `"ps_car_01_cat, ps_ind_12_bin, ps_car_05_cat, ps_reg_03, ps_ind_08_bin, ps_calc_07, ps_calc_02, ps_calc_09, ps_calc_06, ps_calc_08, ps_car_14, ps_ind_11_bin, ps_calc_20_bin, ps_car_15, ps_calc_11, ps_car_09_cat, ps_calc_17_bin, ps_car_11_cat, ps_ind_18_bin, ps_ind_07_bin, ps_reg_01, ps_ind_06_bin, ps_car_02_cat, ps_car_08_cat, ps_calc_14, ps_calc_15_bin, ps_ind_03, ps_car_06_cat, ps_calc_04, ps_ind_17_bin, ps_ind_02_cat, ps_car_12, ps_car_07_cat"`
* cv_score: 0.6284632
* PLB: 0.257 (pretty terrible)

**best single model using logloss after 35 iters**

* level01 - 34
* cv_score: 0.153
* PBL: 0.254


**best single model using error after 35 iters**

* level01 - 1
* cv_score: 0.0364428
* PBL: 0.195

