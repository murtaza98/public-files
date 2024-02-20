try:
    import sys
    from sklearn.metrics import roc_auc_score
    import pandas as pd
    actual_file = sys.argv[1]
    pred_file = sys.argv[2]

    actual_data = pd.read_csv(actual_file)
    actual_data.columns= ['id', 'actual']
    print("Actual Shape:", actual_data.shape)
    # actuals=actual_data['power_output'].tolist()

    pred_data = pd.read_csv(pred_file)
    pred_data.columns= ['id', 'prediction']
    print("Submission Shape:",pred_data.shape)
    # preds=pred_data['power_output'].tolist()

    final = actual_data.merge(pred_data, how='left').fillna(0)
    # print(final.shape)

    metric = roc_auc_score(final['actual'],final['prediction'])

    print(f"FS_SCORE:{metric*100} %")
except Exception as e:
    print("FS_SCORE:0 %")
    print(e)
