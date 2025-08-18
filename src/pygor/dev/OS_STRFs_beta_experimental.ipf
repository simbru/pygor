#pragma rtGlobals=3		// Use modern global access method and strict wave access.

/////////////////////////////////////////////////////////////////////////////////////////////////////
///	Official ScanM Data Preprocessing Scripts - by Tom Baden    	///
/////////////////////////////////////////////////////////////////////////////////////////////////////
///	Requires "ROIs", detrended data stack + trigger stack		///
///	Input Arguments - data Ch (0,1,2...?), Trigger Ch (0,1,2)     	///
///	e.g. "OS_TracesAndTriggers(0,2)"							///
///   --> reads wDataChX_detrended,wDataChY					///
///   --> generates 4 output waves								///
///   - TracesX: (per ROI, raw traces, by frames)					///
///   - TracesX_znorm: (per ROI, z-normalised traces, by frames)	///
///	- TracetimesX: for each frame (per ROI, 2 ms precision)		///
///   - Triggertimes: Timestamps of Triggers (2 ms precision)		///
///   - Triggervalue: Level of each Trigger  event					///
/////////////////////////////////////////////////////////////////////////////////////////////////////

function OS_STRFs_beta_experimental()
	wave OS_parameters
	wave noise_jitter
	
	// STA temporal window parameters (in seconds) - Simen Bruoygard
	variable STA_past_window = 2      // How far into the past to calculate STA (seconds)  
	variable STA_future_window = 2    // How far into the future to calculate STA (seconds)
	
	// Validate STA window parameters
	if (STA_past_window <= 0 || STA_future_window < 0)
		print "ERROR: STA window parameters must be positive (past > 0, future >= 0)"
		abort
	endif
	
	if (STA_past_window + STA_future_window > 10)
		print "WARNING: Very large STA window (", STA_past_window + STA_future_window, "s) may cause memory issues"
	endif
	
	// Performance optimization parameters
	variable use_parallel_processing = 1  // Enable parallel processing where possible
	variable reuse_temp_waves = 1        // Reuse temporary waves to reduce memory allocation
	
	// OS params additions by Simen Bruoygard
	// Handle annoying array naming blunder that I am too lazy to go back and fix - Simen
	if (waveexists(noise_jitter) == 1)
		print "Identified noise_jitter array, renaming to noisearray3d for processing"
		rename noise_jitter, noisearray3d
	endif

	// Set some default values for backwards compatability, in case vals are not in OS_parameters  - Simen
	variable nColours
	if (finddimlabel(OS_parameters, 0, "nColourChannels") == -2)
		nColours = 4
		print "nColourChannels not found in OS_parameters, defaulting to nColours ==", nColours
	else
		nColours = OS_parameters[%nColourChannels]
		print "nColours:", nColours
	endif
		
	variable nTriggers_per_Colour 
	if (finddimlabel(OS_parameters, 0, "STRF_nTriggers_per_Colour") == -2)
		nTriggers_per_Colour = 100
		print "STRF_nTriggers_per_Colour not found in OS_parameters, defaulting to nTriggers_per_Colour =", nTriggers_per_Colour
	else 
		nTriggers_per_Colour = OS_parameters[%STRF_nTriggers_per_Colour]
		print "nTriggers_per_Colour", nTriggers_per_Colour
	endif

	variable CropNoiseEdges
	if (finddimlabel(OS_parameters, 0, "STRF_edge_crop") == -2)
		CropNoiseEdges = 2 //20 for 25um(=1.18deg) // HACK FOR NOW
		print "STRF_edge_crop not found in OS_parameters, defaulting to CropNoiseEdges =", CropNoiseEdges
	else
		CropNoiseEdges = OS_parameters[%STRF_edge_crop]
		print "CropNoiseEdges:", CropNoiseEdges
	endif
	
	variable nF_Max_per_Noiseframe
	if (finddimlabel(OS_parameters, 0, "nF_Max_per_Noiseframe") == -2)
		nF_Max_per_Noiseframe = 8 // how many frames between triggers is allowed as a noise frame
		print "nF_Max_per_Noiseframe not found in OS_parameters, defaulting to nF_Max_per_Noiseframe =", nF_Max_per_Noiseframe
	else
		nF_Max_per_Noiseframe = OS_parameters[%nF_Max_per_Noiseframe]
		print "nF_Max_per_Noiseframe", nF_Max_per_Noiseframe
	endif
	
	// Display and processing constants
	variable RGB_Attenuation = 20 // the larger, the more attenuated the RGB RFs will come out
	variable RGB_Scale_Factor = 2^15-1 // 16-bit scaling factor for RGB display
	variable RGB_Max_Value = 2^16-1    // Maximum RGB value (16-bit)
	variable Default_ROIs_Per_Row = 20 // Default ROIs per row in montage display
	
	variable preSDProjectSmooth = 0
	variable adjust_by_pols = 1

// Dependency checks with improved error handling
// 0 //  check if NoiseArray3D is there
if (waveexists($"NoiseArray3D")==0)
	print "ERROR: NoiseArray3D wave missing - please import! Procedure aborted."
	abort
endif

// 1 // check for Parameter Table
if (waveexists($"OS_Parameters")==0)
	print "Warning: OS_Parameters wave not yet generated - doing that now..."
	OS_ParameterTable()
	DoUpdate
endif
wave OS_Parameters

// 2 //  check for Detrended Data stack
variable Channel = OS_Parameters[%Data_Channel]
if (waveexists($"wDataCh"+Num2Str(Channel)+"_detrended")==0)
	print "Warning: wDataCh"+Num2Str(Channel)+"_detrended wave not yet generated - doing that now..."
	OS_DetrendStack()
endif

// 3 //  check for ROI_Mask
if (waveexists($"ROIs")==0)
	print "Warning: ROIs wave not yet generated - doing that now (using correlation algorithm)..."
	OS_AutoRoiByCorr()
	DoUpdate
endif

// 4 //  check if Traces and Triggers are there
if (waveexists($"Triggertimes")==0)
	print "Warning: Traces and Trigger waves not yet generated - doing that now..."
	OS_TracesAndTriggers()
	DoUpdate
endif

// flags from "OS_Parameters"
variable Display_Stuff = OS_Parameters[%Display_Stuff]
variable use_znorm = OS_Parameters[%Use_Znorm]
variable LineDuration = OS_Parameters[%LineDuration]
variable Noise_PxSize = OS_Parameters[%Noise_PxSize_degree]
variable Event_SD = OS_Parameters[%Noise_EventSD]
variable FilterLength = OS_Parameters[%Noise_FilterLength_s] // Keep for backward compatibility
variable Skip_First_Trig = OS_Parameters[%Skip_First_Triggers]
variable Skip_Last_Trig = OS_Parameters[%Skip_Last_Triggers]

// data handling
wave ROIs
wave Triggertimes_frame // loading by frame, not by time!

// Count triggers more efficiently
variable nTriggers = 0
variable xx,yy,ff,rr,tt,ii,ll  // Main loop variables
variable currentstartframe, currentendframe  // Frame boundary variables
for (tt=0; tt<DimSize(triggertimes_frame,0); tt+=1)
	if (Numtype(triggertimes_frame[tt])==0)
		nTriggers+=1
	else
		break
	endif	
endfor
print nTriggers, "Triggers found"

// Validate trigger skipping parameters
if (Skip_First_Trig < 0 || Skip_Last_Trig < 0)
	print "ERROR: Skip trigger parameters cannot be negative"
	abort
endif

if (Skip_First_Trig + Skip_Last_Trig >= nTriggers)
	print "ERROR: Skip parameters would eliminate all triggers"
	abort
endif

// Identify first and last trigger and select accordingly
variable firsttrigger_f = (Skip_First_Trig != 0) ? triggertimes_frame[Skip_First_Trig] : triggertimes_frame[0]
variable lasttrigger_f = (Skip_Last_Trig != 0) ? triggertimes_frame[nTriggers-Skip_Last_Trig-1] : triggertimes_frame[nTriggers-1]

if (Skip_Last_Trig != 0 || Skip_First_Trig != 0)
	print "Trigger adjustment in OS Parameters detected:"
	print "First trigger specified:", Skip_First_Trig 
	print "Last trigger:", Skip_Last_Trig 
	print "First trigger frame:", firsttrigger_f 
	print "Last trigger frame:", lasttrigger_f 
	print "Triggers after excluding first/last:", nTriggers - Skip_First_Trig - Skip_Last_Trig
endif

// Duplicate temporal data waves to not change the originals 
string tracetimes_name = "Tracetimes"+Num2Str(Channel)
string input_name = "wDataCh"+Num2Str(Channel)+"_detrended"
string traces_name = "Traces"+Num2Str(Channel)+"_raw"
if (use_znorm==1)
	traces_name = "Traces"+Num2Str(Channel)+"_znorm"
endif
duplicate /o $input_name InputStack
duplicate /o $traces_name InputTraces
duplicate /o $tracetimes_name InputTraceTimes

// Calculate frame parameters
variable nF = DimSize(InputTraces,0)
variable nRois = DimSize(InputTraces,1)
variable nY = DimSize(InputStack,1)

variable Frameduration = nY * LineDuration 
variable nF_Filter_past = max(1, floor(STA_past_window / Frameduration))   // frames for past window (minimum 1)
variable nF_Filter_future = max(0, floor(STA_future_window / Frameduration)) // frames for future window
variable nF_Filter = nF_Filter_past + nF_Filter_future // total STA window in frames
variable STA_total_window = STA_past_window + STA_future_window // total window in seconds

// Validate temporal window parameters
if (nF_Filter_past >= nF || nF_Filter >= nF)
	print "ERROR: STA window is larger than available data"
	abort
endif

wave NoiseArray3D
variable nX_Noise = DimSize(NoiseArray3D,1) // X and Y flipped relative to mouse
variable nY_Noise = DimSize(NoiseArray3D,0)
variable nZ_Noise = DimSize(NoiseArray3D,2)

// Validate noise array dimensions
if (nX_Noise <= CropNoiseEdges*2 || nY_Noise <= CropNoiseEdges*2)
	print "ERROR: CropNoiseEdges too large for noise array dimensions"
	abort
endif

string output_name_SD = "STRF_SD"+Num2Str(Channel) // 3D wave (x/y/Roi)
string output_name_Corr = "STRF_Corr"+Num2Str(Channel) // 3D wave (x/y/Roi)
string output_name_individual = "STRF"+Num2Str(Channel)+"_" 

// Pre-allocate output arrays
make /o/n=(nX_Noise,nY_Noise*nColours,nRois) Filter_SDs = 0
make /o/n=(nX_Noise,nY_Noise*nColours,nRois) Filter_Pols = 1 // force to 1 (On)
make /o/n=(nX_Noise,nY_Noise*nColours,nRois) Filter_Corrs = 0

// make Stim Array with optimization
printf "Adjusting NoiseArray to Framerate..."
make /B/o/n=(nX_Noise-CropNoiseEdges*2,nY_Noise-CropNoiseEdges*2,nF) NoiseStimulus_Frameprecision = 0.5

variable nLoops = ceil(nTriggers / nZ_Noise)
print nLoops, "Loops detected"

variable TriggerCounter = 0 
// Optimized trigger loop with better bounds checking
for (tt=Skip_First_trig; tt < nTriggers-Skip_Last_trig-1; tt+=1)
	currentstartframe = triggertimes_frame[tt]
	currentendframe = triggertimes_frame[tt+1]
	
	// Bounds checking
	if (currentstartframe < 0 || currentendframe >= nF)
		// Skip this iteration
	else
	
	if (currentendframe-currentstartframe > nF_Max_per_Noiseframe)
		currentendframe = currentstartframe + nF_Max_per_Noiseframe
	endif
	
		// Parallel assignment with bounds checking
		if (currentendframe < nF && TriggerCounter < nZ_Noise)
			Multithread NoiseStimulus_Frameprecision[][][currentstartframe,currentendframe]=NoiseArray3D[q+CropNoiseEdges][p+CropNoiseEdges][Triggercounter]
		endif
		
		TriggerCounter += 1
		if (TriggerCounter >= nZ_Noise)
			TriggerCounter = 0
		endif
	endif
endfor	

print "done."	

// generate a frameprecision lookup of each colour with optimization
variable nColourLoops = ceil(nTriggers / (nColours*nTriggers_per_Colour))
make /o/n=(nF) ColourLookup = NaN
variable start_idx, end_idx  // Declare variables for trigger indexing
variable colour  // Declare colour variable for loops

for (ll=0; ll<nColourLoops; ll+=1)
	for (colour=0; colour<nColours; colour+=1)
		start_idx = ll*(nColours*nTriggers_per_Colour)+colour*nTriggers_per_Colour
		end_idx = ll*(nColours*nTriggers_per_Colour)+(colour+1)*nTriggers_per_Colour-1
		
		// Bounds checking for trigger indices
		if (start_idx >= nTriggers || end_idx >= nTriggers)
			break
		endif
		
		currentstartframe = triggertimes_frame[start_idx]
		currentendframe = triggertimes_frame[end_idx]
		
		// Bounds checking for frame indices
		if (currentstartframe >= 0 && currentendframe < nF && currentstartframe <= currentendframe)
			ColourLookup[currentstartframe,currentendframe] = colour
		endif
	endfor
endfor

// Get Filters with optimization
printf "Calculating kernels for "
printf Num2Str(nRois)
print " ROIs... "

// Calculate frame parameters needed for pre-allocation
variable nF_relevant = triggertimes_frame[nTriggers-Skip_Last_trig-1] - triggertimes_frame[Skip_First_trig]

// Pre-allocate reusable temporary waves for performance
make /o/n=(max(nF, nF_Filter*2-1)) TempWave_Reusable = 0
make /o/n=(nF_Filter) TempFilter_Reusable = 0

// Clear any existing waves that depend on nF_Filter to prevent old data persistence
killwaves/Z STRFs_concatenated, STRFs_concatenated_SMth, ConcatenatedFilter_SD

// Clear individual filter waves that might have old temporal dimensions
variable cleanup_rr, cleanup_colour
for (cleanup_rr=0; cleanup_rr<nRois; cleanup_rr+=1)
	for (cleanup_colour=0; cleanup_colour<nColours; cleanup_colour+=1)
		string cleanup_filter_name = output_name_individual+Num2Str(cleanup_rr)+"_"+Num2Str(cleanup_colour)
		killwaves/Z $cleanup_filter_name
	endfor
endfor

make /o/n=(nX_Noise,nY_Noise*nColours,nF_Filter*nRois) STRFs_concatenated = NaN

make /o/n=(nRois) eventcounter = 0
make /o/n=(nX_Noise, nY_Noise, nColours) MeanStim = NaN

variable baseline_points, total_corr, neighbor_count, dx, dy  // Declare loop and calculation variables
variable x_start, x_end, y_start, y_end, median_val  // Declare montage variables

for (rr=-1; rr<nRois; rr+=1) // goes through all ROIs
	
	if (rr==-1) // rr == -1 is the reference filter computed as random
		eventcounter = 1000 // meaningless
	else
		// Optimized event counting with reusable waves
		make /o/n=(nF_relevant) CurrentTrace = InputTraces[triggertimes_frame[Skip_First_trig]+p][rr] 
		Differentiate CurrentTrace/D=CurrentTrace_DIF
		
		// Use first 100 points for baseline if available
		baseline_points = min(100, nF_relevant)
		make /o/n=(baseline_points) CurrentTrace_DIFBase = CurrentTrace_DIF[p]
		WaveStats/Q CurrentTrace_DIFBase
		
		if (V_SDev > 0) // Avoid division by zero
			CurrentTrace_DIF -= V_Avg
			CurrentTrace_DIF /= V_SDev	
			
			for (ff=0; ff<nF_relevant; ff+=1)
				if (CurrentTrace_DIF[ff] > Event_SD)
					eventcounter[rr] += 1
				endif
			endfor
		endif
	endif

	if (rr==-1)
		printf "Computing mean stimulus for normalization: Colours..."
	else
		printf "ROI#"+Num2Str(rr+1)+"/"+Num2Str(nRois)+": Colours..."
	endif
	make /o/n=(nF_relevant) CurrentLookup = ColourLookup[triggertimes_frame[Skip_First_trig]+p]
	
	// Optimize: Extract ROI trace once per ROI (outside color loop)
	if (rr==-1)
		make /o/n=(nF_relevant) BaseTrace = 1  // Reference uses constant trace
	else
		make /o/n=(nF_relevant) BaseTrace = InputTraces[triggertimes_frame[Skip_First_trig]+p][rr]
	endif
	
	for (colour=0; colour<nColours; colour+=1)
		// Create color-filtered trace from base trace
		make /o/n=(nF_relevant) CurrentTrace = BaseTrace[p]
		Multithread CurrentTrace[] = (CurrentLookup[p]==colour) ? CurrentTrace[p] : 0
		
		make /o/n=(nX_Noise, nY_Noise, nF_Filter) CurrentFilter = 0
		setscale/p z,-STA_past_window,STA_total_window/nF_Filter,"s" CurrentFilter
		printf Num2Str(colour)
		DoUpdate
		
		if (rr==-1) // compute meanimage with bounds checking
			for (xx=CropNoiseEdges; xx<nX_Noise-CropNoiseEdges; xx+=1)
				for (yy=CropNoiseEdges; yy<nY_Noise-CropNoiseEdges; yy+=1)
					make /o/n=(nF_relevant) CurrentPX = NoiseStimulus_Frameprecision[xx-CropNoiseEdges][yy-CropNoiseEdges][triggertimes_frame[Skip_First_trig]+p] * CurrentTrace[p]
					WaveStats/Q CurrentPX
					MeanStim[xx][yy][colour] = V_Avg
				endfor
			endfor
		else // compute filter with optimization
			for (xx=CropNoiseEdges; xx<nX_Noise-CropNoiseEdges; xx+=1)
				for (yy=CropNoiseEdges; yy<nY_Noise-CropNoiseEdges; yy+=1)
					make /o/n=(nF_relevant) CurrentPX = NoiseStimulus_Frameprecision[xx-CropNoiseEdges][yy-CropNoiseEdges][triggertimes_frame[Skip_First_trig]+p]
					Correlate/NODC CurrentTrace, CurrentPX
					
					// Extract STA: center around trigger time, with past/future windows
					// Improved indexing with bounds checking
					start_idx = nF_relevant - nF_Filter_past
					if (start_idx >= 0 && start_idx + nF_Filter <= DimSize(CurrentPX,0))
						Multithread CurrentFilter[xx][yy][] = CurrentPX[r + start_idx]
					endif
				endfor
			endfor
			
			// Normalize by mean stimulus with safety check
			Multithread CurrentFilter[][][] /= (MeanStim[p][q][colour] != 0) ? MeanStim[p][q][colour] : 1
			Multithread CurrentFilter[][][] = (NumType(CurrentFilter[p][q][r])==2) ? 0 : CurrentFilter[p][q][r] // kill NANs
			
			// Store in concatenated array with bounds checking
			if (rr >= 0 && rr < nRois)
				STRFs_concatenated[][nY_Noise*colour,nY_Noise*(colour+1)-1][nF_Filter*rr,nF_Filter*(rr+1)-1] = CurrentFilter[p][q-nY_Noise*colour][r-nF_Filter*rr]
			endif
			
			// Optimized correlation map calculation with reduced temporary wave creation
			for (xx=CropNoiseEdges; xx<nX_Noise-CropNoiseEdges; xx+=1)
				for (yy=CropNoiseEdges; yy<nY_Noise-CropNoiseEdges; yy+=1)
					// Bounds checking for neighbor pixels
					if (xx > 0 && xx < nX_Noise-1 && yy > 0 && yy < nY_Noise-1)
						make /o/n=(nF_Filter) centerpixel = CurrentFilter[xx][yy][p]
						
						// Calculate neighbor correlations more efficiently
						total_corr = 0
						neighbor_count = 0
						
						// Check 8 neighbors with bounds checking
						variable neighbor_x, neighbor_y  // neighbor pixel coordinates
						for (dx=-1; dx<=1; dx+=1)
							for (dy=-1; dy<=1; dy+=1)
								if (dx==0 && dy==0)
									// skip center pixel
								else
									neighbor_x = xx + dx
									neighbor_y = yy + dy
									if (neighbor_x >= 0 && neighbor_x < nX_Noise && neighbor_y >= 0 && neighbor_y < nY_Noise)
										make /o/n=(nF_Filter) neighbor_px = CurrentFilter[neighbor_x][neighbor_y][p]
										Correlate/NODC centerpixel, neighbor_px
										WaveStats/Q neighbor_px
										total_corr += abs(V_max)
										neighbor_count += 1
									endif
								endif
							endfor
						endfor
						
						if (neighbor_count > 0)
							Filter_Corrs[xx][yy+colour*nY_Noise][rr] = total_corr / neighbor_count
						endif
					endif
				endfor
			endfor		

			// Optimized SD projections calculation
			duplicate /o CurrentFilter CurrentFilter_Smth
			if (preSDProjectSmooth > 0)
				Smooth /Dim=0 preSDProjectSmooth, CurrentFilter_Smth
				Smooth /Dim=1 preSDProjectSmooth, CurrentFilter_Smth
			endif
			
			// z-normalise based on 1st frame with safety check
			make /o/n=(nX_Noise,nY_Noise) tempwave = CurrentFilter_Smth[p][q][0]
			ImageStats/Q tempwave
			if (V_SDev > 0)
				Multithread CurrentFilter_Smth[][][] -= V_Avg
				Multithread CurrentFilter_Smth[][][] /= V_SDev
				
				// compute SD as well as polarity mask
				for (xx=0; xx<nX_Noise; xx+=1)
					for (yy=0; yy<nY_Noise; yy+=1)
						make /o/n=(nF_Filter) CurrentTrace = CurrentFilter_Smth[xx][yy][p]
						WaveStats/Q CurrentTrace 
						if (V_maxloc < V_minloc) // default is On, so here force to Off 
							Filter_Pols[xx][yy+colour*nY_Noise][rr] = -1
						endif
						Filter_SDs[xx][yy+colour*nY_Noise][rr] = V_SDev
					endfor
				endfor
			endif
			
			string filter_name = output_name_individual+Num2Str(rr)+"_"+Num2Str(colour)
			duplicate /o CurrentFilter $filter_name 
		endif		
	endfor // colour loop end
	print "."
endfor

// Optimized correlation map adjustment
variable nX_NoiseCrop = nX_Noise - CropNoiseEdges*2
variable nY_NoiseCrop = nY_Noise - CropNoiseEdges*2

for (rr=0; rr<nRois; rr+=1)
	make /o/n=(nX_NoiseCrop,nY_NoiseCrop*nColours) currentCorr = Filter_Corrs[p+CropNoiseEdges][q][rr]
	Redimension /n=(nX_NoiseCrop*nY_NoiseCrop*nColours) currentCorr	
	median_val = StatsMedian(currentCorr)
	if (median_val != 0)
		Filter_Corrs[][][rr] /= median_val
	endif
endfor

// Apply polarity adjustment if requested
if (adjust_by_pols==1)
	Multithread Filter_Corrs[][][] *= Filter_Pols[p][q][r]
	Multithread Filter_SDs[][][] *= Filter_Pols[p][q][r]
endif

// Optimized concatenated filter SD computation
duplicate /o STRFs_concatenated STRFs_concatenated_SMth
if (preSDProjectSmooth > 0)
	Smooth /Dim=0 preSDProjectSmooth, STRFs_concatenated_SMth
	Smooth /Dim=1 preSDProjectSmooth, STRFs_concatenated_SMth
endif

// z-normalise based on 1st frame
make /o/n=(nX_Noise,nY_Noise) tempwave = STRFs_concatenated_SMth[p][q][0]
ImageStats/Q tempwave
if (V_SDev > 0)
	STRFs_concatenated_SMth[][][] -= V_Avg
	STRFs_concatenated_SMth[][][] /= V_SDev
	
	make /o/n=(nX_Noise,nY_Noise*nColours) ConcatenatedFilter_SD = NaN
	for (xx=0; xx<nX_Noise; xx+=1)
		for (yy=0; yy<nY_Noise*nColours; yy+=1)
			make /o/n=(nF_Filter*nRois) CurrentTrace = STRFs_concatenated_SMth[xx][yy][p]
			WaveStats/Q CurrentTrace 
			ConcatenatedFilter_SD[xx][yy] = V_SDev
		endfor
	endfor
endif

// Optimized display montage creation
variable nROIsMax_Display_per_row = Default_ROIs_Per_Row
variable nRows = Ceil(nRois/nROIsMax_Display_per_row)
variable nColumns = (nRows==1) ? nRois : nROIsMax_Display_per_row

make /o/n=(nColumns*nX_Noise,nRows*nY_Noise*nColours) STRF_Corr_Montage = NaN
variable currentXCoordinate = 0
variable currentYCoordinate = 0

for (rr=0; rr<nRois; rr+=1)
	x_start = currentXCoordinate*nX_Noise
	x_end = (currentXCoordinate+1)*nX_Noise-1
	y_start = currentYCoordinate*nY_Noise*nColours
	y_end = (currentYCoordinate+1)*nY_Noise*nColours-1
	
	STRF_Corr_Montage[x_start,x_end][y_start,y_end] = Filter_Corrs[p-currentXCoordinate*nX_Noise][q-currentYCoordinate*nY_Noise*nColours][rr]
	
	currentXCoordinate += 1
	if (currentXCoordinate >= nColumns)
		currentXCoordinate = 0
		currentYCoordinate += 1
	endif
endfor

// Optimized RGB montage creation
make /o/n=(nColumns*nX_Noise,nRows*nY_Noise,3) STRF_Corr_Montage_RGB = NaN
currentXCoordinate = 0
currentYCoordinate = 0

for (rr=0; rr<nRois; rr+=1)
	x_start = currentXCoordinate*nX_Noise
	x_end = (currentXCoordinate+1)*nX_Noise-1
	y_start = currentYCoordinate*nY_Noise
	y_end = (currentYCoordinate+1)*nY_Noise-1
	
	// Red channel (colour 0)
	if (nColours > 0)
		STRF_Corr_Montage_RGB[x_start,x_end][y_start,y_end][0] = Filter_Corrs[p-currentXCoordinate*nX_Noise][q-currentYCoordinate*nY_Noise][rr]
	endif
	// Green channel (colour 1)  
	if (nColours > 1)
		STRF_Corr_Montage_RGB[x_start,x_end][y_start,y_end][1] = Filter_Corrs[p-currentXCoordinate*nX_Noise][q-currentYCoordinate*nY_Noise+nY_Noise*1][rr]
	endif
	// Blue channel (colour 3 - UV) - only if 4 or more colors available
	if (nColours > 3)
		STRF_Corr_Montage_RGB[x_start,x_end][y_start,y_end][2] = Filter_Corrs[p-currentXCoordinate*nX_Noise][q-currentYCoordinate*nY_Noise+nY_Noise*3][rr]
	endif
	
	currentXCoordinate += 1
	if (currentXCoordinate >= nColumns)
		currentXCoordinate = 0
		currentYCoordinate += 1
	endif
endfor

// Create absolute value version and apply scaling
duplicate/o STRF_Corr_Montage_RGB STRF_Corr_Montage_RGB2
Multithread STRF_Corr_Montage_RGB2[][][] = abs(STRF_Corr_Montage_RGB[p][q][r])

// Apply RGB scaling with bounds checking
Multithread STRF_Corr_Montage_RGB[][][] /= RGB_Attenuation
STRF_Corr_Montage_RGB += 1
Multithread STRF_Corr_Montage_RGB *= RGB_Scale_Factor
Multithread STRF_Corr_Montage_RGB[][][] = (STRF_Corr_Montage_RGB[p][q][r] < 0) ? 0 : STRF_Corr_Montage_RGB[p][q][r]
Multithread STRF_Corr_Montage_RGB[][][] = (STRF_Corr_Montage_RGB[p][q][r] > RGB_Max_Value) ? RGB_Max_Value : STRF_Corr_Montage_RGB[p][q][r]

Multithread STRF_Corr_Montage_RGB2[][][] /= RGB_Attenuation
Multithread STRF_Corr_Montage_RGB2 *= RGB_Scale_Factor
Multithread STRF_Corr_Montage_RGB2[][][] = (STRF_Corr_Montage_RGB2[p][q][r] < 0) ? 0 : STRF_Corr_Montage_RGB2[p][q][r]
Multithread STRF_Corr_Montage_RGB2[][][] = (STRF_Corr_Montage_RGB2[p][q][r] > RGB_Max_Value) ? RGB_Max_Value : STRF_Corr_Montage_RGB2[p][q][r]

print " done."	

// export handling
duplicate /o Filter_Corrs $output_name_Corr
duplicate /o Filter_SDs $output_name_SD

// display
if (Display_Stuff==1)
	// display the Corr montage
	display /k=1
	make /o/n=(1) M_Colors
	ColorTab2Wave Rainbow256
	AppendImage STRF_Corr_Montage
	ModifyGraph fSize=8,noLabel=2,axThick=0
	ModifyImage STRF_Corr_Montage ctab= {-10,10,RedWhiteBlue,0} // 10 means 10 times the median
endif	

// Improved cleanup - only kill waves that definitely exist
killwaves/Z CurrentFilter, InputStack, InputTraces, InputTraceTimes
killwaves/Z CurrentFilter_Smth, tempwave, STRFs_concatenated_SMth
killwaves/Z NoiseStimulus_Frameprecision, ColourLookup, CurrentTrace
killwaves/Z TempWave_Reusable, TempFilter_Reusable
killwaves/Z CurrentPX, CurrentLookup, CurrentTrace_DIF, CurrentTrace_DIFBase
killwaves/Z centerpixel, neighbor_px, currentCorr, BaseTrace

print "OS_STRFs_beta_experimental completed successfully."

end