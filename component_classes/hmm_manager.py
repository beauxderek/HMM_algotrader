import logging
from .data_manager import DataManager
from typing import List, Dict, Tuple
import numpy as np
import os
from .exceptions import ModelUpdateError, InsufficientDataError
from hmmlearn.hmm import GaussianHMM
from numba import jit
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
            
class HMMManager:
    def __init__(self, symbols: List[str], data_manager: DataManager, models_dir: str = "models"):
        self.symbols = symbols
        self.data_manager = data_manager
        self.models_dir = models_dir
        self.models: Dict[str, GaussianHMM] = {}
        self.forward_probs: Dict[str, np.ndarray] = {}
        self.last_reset_date = None

    async def initialize(self, timeframe: str):
        """Load trained models and initialize with historical data"""
        for symbol in self.symbols:
            try:
                # Load trained model
                model_path = os.path.join(self.models_dir, f"{symbol}_hmm_model.pkl")
                if not os.path.exists(model_path):
                    raise ModelUpdateError(f"Model file not found for {symbol}")
                
                with open(model_path, 'rb') as f:
                    self.models[symbol] = pickle.load(f)
                
                # Initialize forward probabilities with historical data
                hist_data = await self.data_manager.get_historical_data(symbol, timeframe)
                if len(hist_data) < self.data_manager.min_history_points:
                    raise InsufficientDataError(
                        f"Insufficient historical data for {symbol}. "
                        f"Need {self.data_manager.min_history_points} points, got {len(hist_data)}"
                    )
                
                self.forward_probs[symbol] = self._compute_forward_probs(
                    symbol,
                    hist_data['returns'].values.reshape(-1, 1)
                )
                logger.info(f"Initialized model and probabilities for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to initialize {symbol}: {str(e)}")
                raise

    def _compute_forward_probs(self, symbol: str, observations: np.ndarray) -> np.ndarray:
        """Compute forward probabilities for a sequence of observations"""
        model = self.models[symbol]
        log_probs = model._compute_log_likelihood(observations)
        forward_probs = np.zeros((len(observations), model.n_components))
        
        # Initialize
        forward_probs[0] = model.startprob_ * np.exp(log_probs[0])
        forward_probs[0] /= forward_probs[0].sum()
        
        # Forward algorithm
        for t in range(1, len(observations)):
            for j in range(model.n_components):
                forward_probs[t, j] = np.sum(
                    forward_probs[t-1] * model.transmat_[:, j]
                ) * np.exp(log_probs[t, j])
            forward_probs[t] /= forward_probs[t].sum()
        
        return forward_probs[-1]

    async def update_probabilities(self, symbol: str, new_return: float):
        """Update forward probabilities with new market data"""
        try:
            model = self.models[symbol]
            observation = np.array([[new_return]])
            log_prob = model._compute_log_likelihood(observation)[0]
            
            # Update forward probabilities
            updated_probs = np.dot(self.forward_probs[symbol], model.transmat_) * np.exp(log_prob)
            self.forward_probs[symbol] = updated_probs / updated_probs.sum()
            
        except Exception as e:
            raise ModelUpdateError(f"Failed to update probabilities for {symbol}: {str(e)}")

    def get_state_probabilities(self, symbol: str) -> Tuple[float, float]:
        """Get current bull and bear probabilities"""
        if symbol not in self.forward_probs:
            raise ModelUpdateError(f"No probabilities available for {symbol}")
        
        bull_state = np.argmax(self.models[symbol].means_.flatten())
        bull_prob = self.forward_probs[symbol][bull_state]
        return bull_prob, 1 - bull_prob
    
@jit(nopython=True)
def multivariate_normal_pdf(self, x, mean, cov):
    """
    Calculate the probability density function of a multivariate normal distribution.
    """
    n = mean.shape[0]
    diff = x - mean
    return (1. / (np.sqrt((2 * np.pi)**n * np.linalg.det(cov))) * 
            np.exp(-0.5 * diff.dot(np.linalg.inv(cov)).dot(diff)))

@jit(nopython=True)
def compute_forward_probabilities_numba(self, startprob, transmat, means, covars, observations):
    n_samples, n_components = len(observations), len(startprob)
    forward_probs = np.zeros((n_samples, n_components))

    for t in range(n_samples):
        if t == 0:
            for j in range(n_components):
                forward_probs[t, j] = startprob[j] * multivariate_normal_pdf(observations[t], means[j], covars[j])
        else:
            for j in range(n_components):
                forward_probs[t, j] = np.sum(forward_probs[t-1] * transmat[:, j]) * multivariate_normal_pdf(observations[t], means[j], covars[j])
        forward_probs[t] /= np.sum(forward_probs[t])

    return forward_probs