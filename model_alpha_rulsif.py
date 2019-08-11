import numpy as np
from scipy import linalg

class Alpha_RULSIF(object):

	def __compmedDist(self, X):
		size1=X.shape[0]
		Xmed = X

		G = np.sum((Xmed*Xmed),1)
		Q = np.tile(G[:, np.newaxis],(1,size1))
		R = np.tile(G,(size1,1))

		dists = Q + R - 2*np.dot(Xmed,Xmed.T)
		dists = dists-np.tril(dists)
		dists= dists.reshape(size1**2,1,order='F').copy()
		return np.sqrt(0.5*np.median(dists[dists>0]))


	def __get_sigma_list(self, x_nu, x_de):
		x = np.c_[x_nu, x_de]
		med = self.__compmedDist(x.T)
		return med*np.array([0.6, 0.8, 1, 1.2, 1.4])


	def __kernel_Gaussian(self, x, c, sigma):
		(d ,nx) = x.shape
		(d ,nc) = c.shape
		x2 = np.sum( x **2 ,0)
		c2 = np.sum( c **2, 0)

		distance2 = np.tile(c2 ,(nx ,1)) + np.tile(x2[:, np.newaxis] ,(1 ,nc)) - 2* np.dot(x.T, c)

		return np.exp(-distance2 / (2 * (sigma ** 2)))


	def R_ULSIF(self, x_nu, x_de, alpha):
		# x_nu: samples from numerator
		# x_de: samples from denominator
		# x_re: reference sample
		# alpha: alpha defined in relative density ratio
		# sigma_list, lambda_list: parameters for model selection
		# b: number of kernel basis
		# fold: number of fold for cross validation

		sigma_list = self.__get_sigma_list(x_nu, x_de)
		lambda_list = np.array(map(lambda x: 10.0**x, np.array([-3, -2, -1, 0, 1])))
		fold = 5
		b = x_nu.shape[1]

		(d, n_nu) = x_nu.shape
		(d, n_de) = x_de.shape
		rand_index = np.random.permutation(n_nu)
		b = min(b, n_nu)
		# x_ce = x_nu[:,rand_index[0:b]]
		x_ce = x_nu[:, np.r_[0:b]]

		score_cv = np.zeros((len(sigma_list), len(lambda_list)))

		cv_index_nu = np.random.permutation(n_nu)
		# cv_index_nu = r_[0:n_nu]
		cv_split_nu = np.floor(np.r_[0:n_nu] * fold / n_nu)
		cv_index_de = np.random.permutation(n_de)
		# cv_index_de = r_[0:n_de]
		cv_split_de = np.floor(np.r_[0:n_de] * fold / n_de)

		for sigma_index in np.r_[0:len(sigma_list)]:
			sigma = sigma_list[sigma_index]
			K_de = self.__kernel_Gaussian(x_de, x_ce, sigma).T
			K_nu = self.__kernel_Gaussian(x_nu, x_ce, sigma).T

			score_tmp = np.zeros((fold, len(lambda_list)))

			for k in np.r_[0:fold]:
				Ktmp1 = K_de[:, cv_index_de[cv_split_de != k]]
				Ktmp2 = K_nu[:, cv_index_nu[cv_split_nu != k]]

				Ktmp = alpha / Ktmp2.shape[1] * np.dot(Ktmp2, Ktmp2.T) + (1 - alpha) / Ktmp1.shape[1] * np.dot(Ktmp1,
																											   Ktmp1.T)
				mKtmp = np.mean(K_nu[:, cv_index_nu[cv_split_nu != k]], 1)

				for lambda_index in np.r_[0: len(lambda_list)]:
					lbd = lambda_list[lambda_index]

					thetat_cv = linalg.solve(Ktmp + (lbd * np.eye(b)), mKtmp)
					thetah_cv = thetat_cv

					score_tmp[k, lambda_index] = alpha * np.mean(
						np.dot(K_nu[:, cv_index_nu[cv_split_nu == k]].T, thetah_cv) ** 2) / 2. \
												 + (1 - alpha) * np.mean(
						np.dot(K_de[:, cv_index_de[cv_split_de == k]].T, thetah_cv) ** 2) / 2. \
												 - np.mean(np.dot(K_nu[:, cv_index_nu[cv_split_nu == k]].T, thetah_cv))

				score_cv[sigma_index, :] = np.mean(score_tmp, 0)

		score_cv_tmp = score_cv.min(1)
		lambda_chosen_index = score_cv.argmin(1)

		score = score_cv_tmp.min()
		sigma_chosen_index = score_cv_tmp.argmin()

		lambda_chosen = lambda_list[lambda_chosen_index[sigma_chosen_index]]
		sigma_chosen = sigma_list[sigma_chosen_index]

		K_de = self.__kernel_Gaussian(x_de, x_ce, sigma_chosen).T
		K_nu = self.__kernel_Gaussian(x_nu, x_ce, sigma_chosen).T

		coe = alpha * np.dot(K_nu, K_nu.T) / n_nu + \
			  (1 - alpha) * np.dot(K_de, K_de.T) / n_de + \
			  lambda_chosen * np.eye(b)
		var = np.mean(K_nu, 1)

		thetat = linalg.solve(coe, var)
		#    thetat=linalg.lstsq(coe,var)[0]
		#    linalg.cho_factor(coe,overwrite_a=True)
		#    linalg.cho_solve((coe,False), var, overwrite_b=True)
		#    thetat = var

		# thetah=maximum(0,thetat)
		thetah = thetat
		wh_x_de = np.dot(K_de.T, thetah).T
		wh_x_nu = np.dot(K_nu.T, thetah).T

		# K_di = kernel_Gaussian(x_re, x_ce, sigma_chosen).T
		# wh_x_re = np.dot(K_di.T, thetah).T

		wh_x_de[wh_x_de < 0] = 0
		# wh_x_re[wh_x_re < 0] = 0

		PE = np.mean(wh_x_nu) - 1. / 2 * (alpha * np.mean(wh_x_nu ** 2) + (1 - alpha) * np.mean(wh_x_de ** 2)) - 1. / 2

		return wh_x_de

