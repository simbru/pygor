import numpy as np

def mean_triggertimes_ms(self):
        """
        Calculate the mean trigger times in milliseconds.

        Parameters
        ----------
        None

        Returns
        -------
        mean_triggertimes : numpy.ndarray
            The mean trigger times in milliseconds for each phase of the stimulus cycle.
        """
        triggertimes_ms = (self.triggertimes_frame-self.triggertimes_frame[0])/self.frame_hz/self.linedur_s*self.n_planes 
        triggertimes_ms_reshaped = triggertimes_ms[:-1].reshape(-1, self.trigger_mode)
        mean_triggertimes = np.mean(triggertimes_ms_reshaped, axis=0)-np.mean(triggertimes_ms_reshaped, axis=0)[0]
        return mean_triggertimes