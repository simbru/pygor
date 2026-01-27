# """
# Von Mises curve fitting for preferred direction and orientation estimation.

# This module implements von Mises (circular Gaussian) curve fitting for estimating
# preferred direction and orientation from tuning curves, following the approach
# described in Antinucci et al. 2016.

# Key features:
# - Single von Mises fitting for direction-selective (DS) cells
# - Double von Mises (180° apart) fitting for orientation-selective (OS) cells
# - R² goodness-of-fit filtering for quality control
# - Vectorized batch processing for multiple ROIs

# References:
#     Antinucci, P., et al. (2016). Neural Mechanisms Generating Orientation
#     Selectivity in the Retina. Current Biology.
# """

# import numpy as np
# from scipy.optimize import curve_fit
# from scipy.special import i0


# # =============================================================================
# # Von Mises Model Functions
# # =============================================================================

# def von_mises_single(theta, mu, kappa, amplitude, baseline):
#     """
#     Single von Mises function for direction selectivity.

#     Models a single-peaked tuning curve centered at mu.

#     Parameters
#     ----------
#     theta : array-like
#         Angles in radians
#     mu : float
#         Preferred direction (center) in radians
#     kappa : float
#         Concentration parameter (higher = narrower tuning, analogous to 1/variance)
#     amplitude : float
#         Peak amplitude above baseline
#     baseline : float
#         Baseline response level

#     Returns
#     -------
#     np.ndarray
#         Response values at each angle
#     """
#     theta = np.asarray(theta)
#     # Von Mises distribution (circular Gaussian)
#     vm = np.exp(kappa * np.cos(theta - mu))
#     # Normalize to [0, 1] range
#     vm_max = vm.max()
#     if vm_max > 0:
#         vm_normalized = vm / vm_max
#     else:
#         vm_normalized = vm
#     return baseline + amplitude * vm_normalized


# def von_mises_double(theta, mu, kappa, amplitude, baseline):
#     """
#     Sum of two von Mises 180 degrees apart for orientation selectivity.

#     Models a double-peaked tuning curve with peaks at mu and mu + π,
#     capturing the axis symmetry of orientation-selective responses.

#     Parameters
#     ----------
#     theta : array-like
#         Angles in radians
#     mu : float
#         Preferred orientation (center) in radians (0 to π)
#     kappa : float
#         Concentration parameter
#     amplitude : float
#         Peak amplitude above baseline
#     baseline : float
#         Baseline response level

#     Returns
#     -------
#     np.ndarray
#         Response values at each angle
#     """
#     theta = np.asarray(theta)
#     # Two von Mises peaks 180 degrees apart
#     vm1 = np.exp(kappa * np.cos(theta - mu))
#     vm2 = np.exp(kappa * np.cos(theta - mu - np.pi))
#     vm_sum = vm1 + vm2
#     # Normalize to [0, 1] range
#     vm_max = vm_sum.max()
#     if vm_max > 0:
#         vm_normalized = vm_sum / vm_max
#     else:
#         vm_normalized = vm_sum
#     return baseline + amplitude * vm_normalized


# # =============================================================================
# # Goodness-of-Fit
# # =============================================================================

# def compute_r_squared(observed, predicted):
#     """
#     Compute coefficient of determination (R-squared).

#     R² = 1 - (SS_res / SS_tot)

#     Where:
#     - SS_res = sum((observed - predicted)²)
#     - SS_tot = sum((observed - mean(observed))²)

#     Parameters
#     ----------
#     observed : array-like
#         Observed response values
#     predicted : array-like
#         Predicted response values from model

#     Returns
#     -------
#     float
#         R-squared value (0 to 1, can be negative for very poor fits)
#     """
#     observed = np.asarray(observed)
#     predicted = np.asarray(predicted)

#     ss_res = np.sum((observed - predicted) ** 2)
#     ss_tot = np.sum((observed - np.mean(observed)) ** 2)

#     if ss_tot == 0:
#         return 0.0  # Constant response
#     return 1 - (ss_res / ss_tot)


# # =============================================================================
# # Initial Parameter Estimation
# # =============================================================================

# def _estimate_initial_params_direction(responses, directions_rad):
#     """
#     Estimate initial parameters for direction von Mises fitting.

#     Parameters
#     ----------
#     responses : array-like
#         Response values for each direction
#     directions_rad : array-like
#         Direction values in radians

#     Returns
#     -------
#     list
#         Initial parameters [mu, kappa, amplitude, baseline]
#     """
#     responses = np.asarray(responses)
#     directions_rad = np.asarray(directions_rad)

#     baseline = np.min(responses)
#     amplitude = np.max(responses) - baseline
#     # Use argmax for initial mu estimate
#     mu = directions_rad[np.argmax(responses)]
#     # Start with moderate concentration
#     kappa = 2.0

#     return [mu, kappa, amplitude, baseline]


# def _estimate_initial_params_orientation(responses, directions_rad):
#     """
#     Estimate initial parameters for orientation von Mises fitting.

#     For orientation, we need to find the orientation axis (0-180°)
#     that best captures the double-peaked response.

#     Parameters
#     ----------
#     responses : array-like
#         Response values for each direction
#     directions_rad : array-like
#         Direction values in radians

#     Returns
#     -------
#     list
#         Initial parameters [mu, kappa, amplitude, baseline]
#     """
#     responses = np.asarray(responses)
#     directions_rad = np.asarray(directions_rad)

#     baseline = np.min(responses)
#     amplitude = np.max(responses) - baseline

#     # For orientation, use doubled-angle method to find initial mu
#     # This handles the 180° periodicity properly
#     theta_2x = directions_rad * 2
#     x = np.sum(responses * np.cos(theta_2x))
#     y = np.sum(responses * np.sin(theta_2x))
#     mu_doubled = np.arctan2(y, x)
#     mu = mu_doubled / 2  # Convert back to orientation space

#     # Ensure mu is in [0, π)
#     if mu < 0:
#         mu += np.pi

#     kappa = 2.0

#     return [mu, kappa, amplitude, baseline]


# # =============================================================================
# # Single ROI Fitting Functions
# # =============================================================================

# def fit_von_mises_direction(responses, directions_deg, return_full=False):
#     """
#     Fit single von Mises to estimate preferred direction.

#     For direction-selective (DS) cells. Finds the center (μ) of a
#     single-peaked von Mises fit to the tuning curve.

#     Parameters
#     ----------
#     responses : array-like
#         Response values for each direction (1D: n_directions)
#     directions_deg : array-like
#         Direction values in degrees (0-360)
#     return_full : bool, optional
#         If True, return full fit results dict.
#         If False, return just (preferred_direction, r_squared).

#     Returns
#     -------
#     If return_full=False:
#         tuple : (preferred_direction_deg, r_squared)
#     If return_full=True:
#         dict : {
#             'preferred_direction': float (degrees, 0-360),
#             'kappa': float (concentration parameter),
#             'amplitude': float,
#             'baseline': float,
#             'r_squared': float,
#             'fit_successful': bool,
#             'fitted_curve': np.ndarray (predicted values at input directions)
#         }
#     """
#     responses = np.asarray(responses, dtype=float)
#     directions_deg = np.asarray(directions_deg, dtype=float)
#     directions_rad = np.deg2rad(directions_deg)

#     # Handle edge cases
#     if len(responses) < 4:
#         return _failed_fit_result(return_full, 'direction')

#     if np.all(responses == responses[0]):  # Constant response
#         return _failed_fit_result(return_full, 'direction')

#     # Initial parameter estimates
#     p0 = _estimate_initial_params_direction(responses, directions_rad)

#     # Fitting bounds
#     # mu: full circle, kappa: positive, amplitude: non-negative, baseline: any
#     bounds = (
#         [-np.pi, 0.1, 0, -np.inf],
#         [np.pi, 100, np.inf, np.inf]
#     )

#     # Attempt fitting
#     try:
#         popt, pcov = curve_fit(
#             von_mises_single, directions_rad, responses,
#             p0=p0, bounds=bounds, maxfev=100
#         )
#         mu, kappa, amplitude, baseline = popt

#         # Compute fitted curve and R²
#         fitted = von_mises_single(directions_rad, *popt)
#         r_squared = compute_r_squared(responses, fitted)

#         # Convert mu to degrees and ensure 0-360 range
#         preferred_direction = np.rad2deg(mu) % 360

#         if return_full:
#             return {
#                 'preferred_direction': preferred_direction,
#                 'kappa': kappa,
#                 'amplitude': amplitude,
#                 'baseline': baseline,
#                 'r_squared': r_squared,
#                 'fit_successful': True,
#                 'fitted_curve': fitted
#             }
#         else:
#             return (preferred_direction, r_squared)

#     except (RuntimeError, ValueError):
#         # Retry disabled for now (use maxfev in the primary fit)
#         return _failed_fit_result(return_full, 'direction')


# def fit_von_mises_orientation(responses, directions_deg, return_full=False):
#     """
#     Fit double von Mises (180° apart) to estimate preferred orientation.

#     For orientation-selective (OS) cells. Fits a sum of two von Mises
#     peaks 180° apart to capture the axis symmetry of orientation tuning.

#     Parameters
#     ----------
#     responses : array-like
#         Response values for each direction (1D: n_directions)
#     directions_deg : array-like
#         Direction values in degrees (0-360)
#     return_full : bool, optional
#         If True, return full fit results dict.
#         If False, return just (preferred_orientation, r_squared).

#     Returns
#     -------
#     If return_full=False:
#         tuple : (preferred_orientation_deg, r_squared)
#     If return_full=True:
#         dict : {
#             'preferred_orientation': float (degrees, 0-180),
#             'kappa': float,
#             'amplitude': float,
#             'baseline': float,
#             'r_squared': float,
#             'fit_successful': bool,
#             'fitted_curve': np.ndarray
#         }
#     """
#     responses = np.asarray(responses, dtype=float)
#     directions_deg = np.asarray(directions_deg, dtype=float)
#     directions_rad = np.deg2rad(directions_deg)

#     # Handle edge cases
#     if len(responses) < 4:
#         return _failed_fit_result(return_full, 'orientation')

#     if np.all(responses == responses[0]):  # Constant response
#         return _failed_fit_result(return_full, 'orientation')

#     # Initial parameter estimates
#     p0 = _estimate_initial_params_orientation(responses, directions_rad)

#     # Fitting bounds - mu constrained to [0, π) for orientation
#     bounds = (
#         [0, 0.1, 0, -np.inf],
#         [np.pi, 100, np.inf, np.inf]
#     )

#     # Attempt fitting
#     try:
#         popt, pcov = curve_fit(
#             von_mises_double, directions_rad, responses,
#             p0=p0, bounds=bounds, maxfev=100
#         )
#         mu, kappa, amplitude, baseline = popt

#         # Compute fitted curve and R²
#         fitted = von_mises_double(directions_rad, *popt)
#         r_squared = compute_r_squared(responses, fitted)

#         # Convert mu to degrees (0-180 range)
#         preferred_orientation = np.rad2deg(mu) % 180

#         if return_full:
#             return {
#                 'preferred_orientation': preferred_orientation,
#                 'kappa': kappa,
#                 'amplitude': amplitude,
#                 'baseline': baseline,
#                 'r_squared': r_squared,
#                 'fit_successful': True,
#                 'fitted_curve': fitted
#             }
#         else:
#             return (preferred_orientation, r_squared)

#     except (RuntimeError, ValueError):
#         # Retry disabled for now (use maxfev in the primary fit)
#         return _failed_fit_result(return_full, 'orientation')


# def _alternative_initial_params(responses, directions_rad, fit_type):
#     """Generate alternative initial parameters for retry (currently unused)."""
#     responses = np.asarray(responses)

#     baseline = np.median(responses)  # Use median instead of min
#     amplitude = np.ptp(responses)  # Peak-to-peak range

#     if fit_type == 'direction':
#         # Use circular mean for mu
#         x = np.sum(responses * np.cos(directions_rad))
#         y = np.sum(responses * np.sin(directions_rad))
#         mu = np.arctan2(y, x)
#     else:  # orientation
#         theta_2x = directions_rad * 2
#         x = np.sum(responses * np.cos(theta_2x))
#         y = np.sum(responses * np.sin(theta_2x))
#         mu = np.arctan2(y, x) / 2
#         if mu < 0:
#             mu += np.pi

#     kappa = 1.0  # Lower kappa for broader fits

#     return [mu, kappa, amplitude, baseline]


# def _failed_fit_result(return_full, fit_type):
#     """Return result for failed fit."""
#     if fit_type == 'direction':
#         key = 'preferred_direction'
#     else:
#         key = 'preferred_orientation'

#     if return_full:
#         return {
#             key: np.nan,
#             'kappa': np.nan,
#             'amplitude': np.nan,
#             'baseline': np.nan,
#             'r_squared': np.nan,
#             'fit_successful': False,
#             'fitted_curve': None
#         }
#     else:
#         return (np.nan, np.nan)


# # =============================================================================
# # Vectorized Batch Functions
# # =============================================================================

# def compute_vonmises_preferred_direction(responses, directions_deg, r_squared_threshold=0.8):
#     """
#     Compute preferred direction via von Mises fitting for multiple ROIs.

#     Parameters
#     ----------
#     responses : array-like
#         Response array. Can be:
#         - 1D (n_directions): single ROI
#         - 2D (n_rois, n_directions): multiple ROIs
#         - 3D (n_phases, n_rois, n_directions): multi-phase data
#     directions_deg : array-like
#         1D array of direction values in degrees (0-360)
#     r_squared_threshold : float, optional
#         Minimum R-squared for valid fit (default 0.8).
#         Fits with R² below this are marked as invalid.

#     Returns
#     -------
#     dict : {
#         'preferred_direction': np.ndarray (degrees, NaN for failed/low-R² fits),
#         'kappa': np.ndarray (concentration parameter),
#         'r_squared': np.ndarray,
#         'fit_valid': np.ndarray (bool mask, True if fit succeeded AND R² >= threshold)
#     }
#     """
#     responses = np.asarray(responses)
#     directions_deg = np.asarray(directions_deg)

#     # Handle different input dimensions
#     if responses.ndim == 1:
#         # Single ROI
#         pref_dir, r_sq = fit_von_mises_direction(responses, directions_deg)
#         fit_valid = not np.isnan(r_sq) and r_sq >= r_squared_threshold
#         return {
#             'preferred_direction': np.array([pref_dir]),
#             'kappa': np.array([np.nan]),  # Would need return_full for kappa
#             'r_squared': np.array([r_sq]),
#             'fit_valid': np.array([fit_valid])
#         }

#     elif responses.ndim == 2:
#         # Multiple ROIs: (n_rois, n_directions)
#         n_rois = responses.shape[0]
#         preferred_directions = np.zeros(n_rois)
#         kappas = np.zeros(n_rois)
#         r_squareds = np.zeros(n_rois)
#         fit_valid = np.zeros(n_rois, dtype=bool)

#         for i in range(n_rois):
#             result = fit_von_mises_direction(responses[i], directions_deg, return_full=True)
#             preferred_directions[i] = result['preferred_direction']
#             kappas[i] = result['kappa']
#             r_squareds[i] = result['r_squared']
#             fit_valid[i] = (result['fit_successful'] and
#                           not np.isnan(result['r_squared']) and
#                           result['r_squared'] >= r_squared_threshold)

#         return {
#             'preferred_direction': preferred_directions,
#             'kappa': kappas,
#             'r_squared': r_squareds,
#             'fit_valid': fit_valid
#         }

#     elif responses.ndim == 3:
#         # Multi-phase: (n_phases, n_rois, n_directions)
#         n_phases, n_rois, _ = responses.shape
#         preferred_directions = np.zeros((n_phases, n_rois))
#         kappas = np.zeros((n_phases, n_rois))
#         r_squareds = np.zeros((n_phases, n_rois))
#         fit_valid = np.zeros((n_phases, n_rois), dtype=bool)

#         for p in range(n_phases):
#             for i in range(n_rois):
#                 result = fit_von_mises_direction(responses[p, i], directions_deg, return_full=True)
#                 preferred_directions[p, i] = result['preferred_direction']
#                 kappas[p, i] = result['kappa']
#                 r_squareds[p, i] = result['r_squared']
#                 fit_valid[p, i] = (result['fit_successful'] and
#                                   not np.isnan(result['r_squared']) and
#                                   result['r_squared'] >= r_squared_threshold)

#         return {
#             'preferred_direction': preferred_directions,
#             'kappa': kappas,
#             'r_squared': r_squareds,
#             'fit_valid': fit_valid
#         }

#     else:
#         raise ValueError(f"responses must be 1D, 2D, or 3D array, got {responses.ndim}D")


# def compute_vonmises_preferred_orientation(responses, directions_deg, r_squared_threshold=0.8):
#     """
#     Compute preferred orientation via double von Mises fitting for multiple ROIs.

#     Parameters
#     ----------
#     responses : array-like
#         Response array. Can be:
#         - 1D (n_directions): single ROI
#         - 2D (n_rois, n_directions): multiple ROIs
#         - 3D (n_phases, n_rois, n_directions): multi-phase data
#     directions_deg : array-like
#         1D array of direction values in degrees (0-360)
#     r_squared_threshold : float, optional
#         Minimum R-squared for valid fit (default 0.8).

#     Returns
#     -------
#     dict : {
#         'preferred_orientation': np.ndarray (degrees 0-180, NaN for failed fits),
#         'kappa': np.ndarray,
#         'r_squared': np.ndarray,
#         'fit_valid': np.ndarray (bool)
#     }
#     """
#     responses = np.asarray(responses)
#     directions_deg = np.asarray(directions_deg)

#     # Handle different input dimensions
#     if responses.ndim == 1:
#         # Single ROI
#         pref_ori, r_sq = fit_von_mises_orientation(responses, directions_deg)
#         fit_valid = not np.isnan(r_sq) and r_sq >= r_squared_threshold
#         return {
#             'preferred_orientation': np.array([pref_ori]),
#             'kappa': np.array([np.nan]),
#             'r_squared': np.array([r_sq]),
#             'fit_valid': np.array([fit_valid])
#         }

#     elif responses.ndim == 2:
#         # Multiple ROIs: (n_rois, n_directions)
#         n_rois = responses.shape[0]
#         preferred_orientations = np.zeros(n_rois)
#         kappas = np.zeros(n_rois)
#         r_squareds = np.zeros(n_rois)
#         fit_valid = np.zeros(n_rois, dtype=bool)

#         for i in range(n_rois):
#             result = fit_von_mises_orientation(responses[i], directions_deg, return_full=True)
#             preferred_orientations[i] = result['preferred_orientation']
#             kappas[i] = result['kappa']
#             r_squareds[i] = result['r_squared']
#             fit_valid[i] = (result['fit_successful'] and
#                           not np.isnan(result['r_squared']) and
#                           result['r_squared'] >= r_squared_threshold)

#         return {
#             'preferred_orientation': preferred_orientations,
#             'kappa': kappas,
#             'r_squared': r_squareds,
#             'fit_valid': fit_valid
#         }

#     elif responses.ndim == 3:
#         # Multi-phase: (n_phases, n_rois, n_directions)
#         n_phases, n_rois, _ = responses.shape
#         preferred_orientations = np.zeros((n_phases, n_rois))
#         kappas = np.zeros((n_phases, n_rois))
#         r_squareds = np.zeros((n_phases, n_rois))
#         fit_valid = np.zeros((n_phases, n_rois), dtype=bool)

#         for p in range(n_phases):
#             for i in range(n_rois):
#                 result = fit_von_mises_orientation(responses[p, i], directions_deg, return_full=True)
#                 preferred_orientations[p, i] = result['preferred_orientation']
#                 kappas[p, i] = result['kappa']
#                 r_squareds[p, i] = result['r_squared']
#                 fit_valid[p, i] = (result['fit_successful'] and
#                                   not np.isnan(result['r_squared']) and
#                                   result['r_squared'] >= r_squared_threshold)

#         return {
#             'preferred_orientation': preferred_orientations,
#             'kappa': kappas,
#             'r_squared': r_squareds,
#             'fit_valid': fit_valid
#         }

#     else:
#         raise ValueError(f"responses must be 1D, 2D, or 3D array, got {responses.ndim}D")


# # =============================================================================
# # Plotting Utilities
# # =============================================================================

# def plot_vonmises_fit(responses, directions_deg, fit_type='direction', ax=None):
#     """
#     Plot tuning curve with von Mises fit overlay.

#     Parameters
#     ----------
#     responses : array-like
#         Response values for each direction (1D)
#     directions_deg : array-like
#         Direction values in degrees
#     fit_type : str
#         'direction' for single von Mises, 'orientation' for double
#     ax : matplotlib axis, optional
#         Axis to plot on. If None, creates new polar axis.

#     Returns
#     -------
#     matplotlib.axes.Axes
#         The axis with the plot
#     dict
#         Fit results
#     """
#     import matplotlib.pyplot as plt

#     responses = np.asarray(responses)
#     directions_deg = np.asarray(directions_deg)

#     # Perform fit
#     if fit_type == 'direction':
#         result = fit_von_mises_direction(responses, directions_deg, return_full=True)
#         angle_key = 'preferred_direction'
#     else:
#         result = fit_von_mises_orientation(responses, directions_deg, return_full=True)
#         angle_key = 'preferred_orientation'

#     # Create axis if needed
#     if ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={'projection': 'polar'})

#     # Sort directions for smooth curve
#     sort_idx = np.argsort(directions_deg)
#     theta = np.deg2rad(directions_deg[sort_idx])
#     r = responses[sort_idx]

#     # Close the curve
#     theta_closed = np.concatenate([theta, [theta[0]]])
#     r_closed = np.concatenate([r, [r[0]]])

#     # Plot data
#     ax.plot(theta_closed, r_closed, 'b-', linewidth=2, label='Data')
#     ax.scatter(theta, r, c='blue', s=50, zorder=5)

#     # Plot fit if successful
#     if result['fit_successful'] and result['fitted_curve'] is not None:
#         fitted_sorted = result['fitted_curve'][sort_idx]
#         fitted_closed = np.concatenate([fitted_sorted, [fitted_sorted[0]]])
#         ax.plot(theta_closed, fitted_closed, 'r--', linewidth=2,
#                 label=f"Fit (R²={result['r_squared']:.2f})")

#         # Mark preferred angle
#         pref_angle_rad = np.deg2rad(result[angle_key])
#         ax.axvline(pref_angle_rad, color='red', linestyle='-', alpha=0.7,
#                    label=f"Pref: {result[angle_key]:.0f}°")

#         if fit_type == 'orientation':
#             # Also mark the opposite orientation
#             ax.axvline(pref_angle_rad + np.pi, color='red', linestyle='-', alpha=0.3)

#     ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

#     return ax, result
