@echo off
setlocal

:: Set the conda environment name
set "ENV_NAME=cristal_dev"

:: Set the path to your project directory (optional but recommended)
set "PROJECT_DIR=%~dp0"

echo Activating conda environment: %ENV_NAME%
echo Project directory: %PROJECT_DIR%

:: Activate conda environment
call conda activate %ENV_NAME%

:: Check if activation was successful
if %errorlevel% neq 0 (
    echo Error: Failed to activate conda environment %ENV_NAME%
    pause
    exit /b 1
)

echo Running unit tests...
echo.

:: Run the unittest command
python -m unittest discover -s tests -p "test_*.py"

:: Capture the exit code
set EXIT_CODE=%errorlevel%

echo.
if %EXIT_CODE% equ 0 (
    echo All tests passed successfully!
) else (
    echo Some tests failed with exit code %EXIT_CODE%
)

pause
exit /b %EXIT_CODE%