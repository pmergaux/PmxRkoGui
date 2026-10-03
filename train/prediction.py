import numpy as np
import tensorflow as tf
import warnings
import pandas as pd

keras_rnn = ['LSTM', 'SIMPLE', 'ULTRA', 'GRU']
no_keras_rnn = ['LGBM', 'XGB', 'CAT', 'TAB', 'TABFF']

def prediction(model, X_test, X_test_seq=None,VERSION=[]):
	"""
	Problème principal : confusion X_test vs X_test_seq + conditions LSTM dupliquées
	"""
	if model is None:
		print(f"Modèle None pour {VERSION}")
		return None

	proba = np.full(len(X_test), 0.5)
	try:
		if any(k in VERSION for k in no_keras_rnn):
			if hasattr(model, "predict_proba"):
				# Classifieurs sklearn/xgboost/lgbm
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", category=UserWarning)
					proba = model.predict_proba(X_test)[:, 1]
			elif hasattr(model, "predict"):
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", category=UserWarning)
					proba = model.predict(X_test)
		elif any(k in VERSION for k in keras_rnn):
			if X_test_seq is None or len(X_test_seq) == 0:
				print(f"[INFO pred] {VERSION} : Données insuffisantes {X_test_seq}")
			else:
				if X_test_seq is not None and X_test_seq.ndim == 3:
					X_in = X_test_seq
				else:
					X_in = X_test.reshape(-1, 1, X_test.shape[-1])
				proba = model.predict(X_in, verbose=0)
		# pour si au-dessus en erreur
		elif 'MLP' in VERSION:
				# Retrait de l'argument 'training=False'
				proba = model.predict(X_test, verbose=0)
				# Si le modèle renvoie un tenseur (ce qui peut arriver avec Keras 3),
				# assurez-vous de convertir en numpy proprement
				if hasattr(proba, 'numpy'):
					proba = proba.numpy()
		if proba is not None:
			# print(f"DEBUG RNN RAW: {proba[-5:]}")  # Affiche les 5 dernières# prédictions brutes
			proba = np.asarray(proba).ravel()  # uniformise

	except Exception as e:
		print(f"→ ÉCHEC {VERSION}: {e.__class__.__name__} → {str(e)[:180]}")
		# traceback.print_exc()   # décommente en debug

	return proba

def tabicl_predict(model, X_test):
	"""
	Inférence de TabICL pour prédire les probabilités (classe positive).
	"""
	try:
		proba = model.predict(X_test)
		if proba is not None:
			"""
			# Appliquer la sigmoïde pour normaliser la régression
			proba_p = 1 / (1 + np.exp(-proba * 100))  # Le *100 aide à mieux séparer les signaux
			if proba_p is not None:
				proba_p = np.asarray(proba_p).ravel()
				proba_p = np.clip(proba_p, 0.001, 0.999)
			"""
			proba = np.asarray(proba).ravel()
			#print(f"DEBUG TAB RAW: {proba[-5:]} probabilité {proba_p[-1]:.3f}")  # Affiche les 5 dernières# prédictions brutes

	except Exception as e:
		print(f"→ ÉCHEC tabicl: {e}")
		return np.zeros(len(X_test))

	return proba

