@echo off
echo Pygor Test Suite
echo ================

REM Navigate to project root
cd /d "%~dp0\..\..\.."

REM Try to activate conda environment if it exists
if exist "%CONDA_PREFIX%" (
    echo Activating conda environment...
    call conda activate strfsclone 2>nul
)

REM Run the comprehensive test suite
echo Running comprehensive test suite...
python src/pygor/test/run_tests.py

echo.
echo Test run complete. Check output above for results.
pause