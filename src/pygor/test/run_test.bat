@echo off
echo Pygor Test Suite
echo ================

REM Navigate to pygor root, where pyproject.toml holds the pytest config
cd /d "%~dp0\..\..\.."

python src/pygor/test/run_tests.py %*

echo.
echo Test run complete. Check output above for results.
pause
