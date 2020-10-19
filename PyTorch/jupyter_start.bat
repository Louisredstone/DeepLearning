@echo off

conda info -e>jupyter_start.tmp.txt | echo 0 1>nul
@REM it's very tricky. You'll need "| echo 0 1>nul", otherwise this batch file
@REM will not go on
@REM maybe because `conda info -e` uses exceptions to quit

set flag=false
@REM if there is env "pytorch"

for /F %%i in (jupyter_start.tmp.txt) do (
    if "%%i"=="pytorch" (
        set flag=true
    )
)

del jupyter_start.tmp.txt 2>nul

if "%flag%"=="true" (
    conda activate pytorch
    jupyter notebook
) else (
    echo ERROR: env "pytorch" does not exist.
)