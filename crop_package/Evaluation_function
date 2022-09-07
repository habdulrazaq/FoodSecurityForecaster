def Evaluation_Function(X, Y):
  res = model.evaluate(X_test_train, y_test_train, verbose=0)
  mae_lstm = res[1]
  list_of_mae_recurrent_model.append(mae_lstm)
  print(f"MAE LSTM fold n°{fold_id} = {round(mae_lstm, 2)}")

  '''Comparison LSTM vs Baseline for the current fold'''
  return print(f"🏋🏽‍♂️ improvement over baseline: {round((1 - (mae_lstm/mae_baseline))*100,2)} % \n")
