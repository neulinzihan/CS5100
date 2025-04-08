@echo off
chcp 65001 >nul

:: Get today's date (YYYY-MM-DD)
for /f %%i in ('powershell -Command "Get-Date -Format yyyy-MM-dd"') do set today=%%i

echo ================================
echo Git Auto Commit Tool
echo ================================

set /p msg=Enter commit message (leave empty to use default with date):

if "%msg%"=="" set msg=Auto commit - %today%

echo.
echo Adding files...
git add .

echo Committing: %msg%
git commit -m "%msg%"

echo Pushing to GitHub...
git push

pause
