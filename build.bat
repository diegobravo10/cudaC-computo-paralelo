@echo off
REM Compila el programa principal y las pruebas.
REM Requiere CUDA Toolkit y Visual Studio (MSVC). Si cl.exe no esta en el PATH,
REM se carga el entorno de Visual Studio con vcvars64.bat.

where cl >nul 2>nul
if not errorlevel 1 goto compilar

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%i in (`call "%VSWHERE%" -latest -prerelease -products * -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
    echo No se encontro Visual Studio. Abra "x64 Native Tools Command Prompt" y vuelva a ejecutar.
    exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul

:compilar
if not exist bin mkdir bin

set FLAGS=-O3 -arch=native -std=c++17 -Xcompiler "/O2 /EHsc /utf-8"
set SRC=src\conv_cpu.cpp src\conv_gpu.cu src\imagen.cpp src\mascaras.cpp

nvcc %FLAGS% src\main.cpp %SRC% -o bin\convolucion.exe || exit /b 1
nvcc %FLAGS% tests\pruebas.cpp %SRC% -o bin\pruebas.exe || exit /b 1

echo Compilacion correcta: bin\convolucion.exe y bin\pruebas.exe
