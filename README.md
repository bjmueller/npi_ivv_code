# How to set up the Python code

## Option 1 - Local Python installation (recommended):
1. The code requires a current Python and Jupyter installation (Python version 3.8 or above recommend). If you have not installed Python already, we recommend [Anaconda](https://www.anaconda.com/) as an all-platform solution.
2. Generate the file `data_main.rData` (input data, see below) and place it in the  same directory as the Python (`.py`) files and the Jupyter notebook `StopptCOVID_IVV.ipynb`.
3. You may need to install packages (using `conda install…`, `pip3 install…`, or a graphical package management system if you have one). Make sure the following packages are installed:
   * numpy
   * matplotlib
   * scipy
   * statsmodels
   * sklearn
   * pandas
   * textwrap
   * tqdm
   * [arch](https://pypi.org/project/arch/)
   * [pyreadr](https://ofajardo.github.io/pyreadr/_build/html/index.html)
4. Open and run the Jupyter notebook `StopptCOVID_IVV.ipynb`.
   * By default, the notebook does not run the entire ARMA model grid because this takes considerable time. Comment the call back in if needed.
   * The notebook will generate all the plots used in the paper.
   * The renewal model and random forest regression may also take a long time to run.
   * Running all the models on the actual data for R(t) and on the synthetic R(t) for the injection recovery test will take several hours. Consider reducing the number of bootstrap samples if you merely want to verify that the code runs and reproduces the point estimates.


## Option 2 - Google Colab and Google Drive:
1. Upload `StopptCOVID_IVV.ipynb` on Google Colab.
2. Upload all of the `.py` files and `data_main.rData` to a folder `IVV-Code-and-Data/` in your Google Drive account.
3. Open and run the Colab notebook. You will be required to give Colab appropriate permissions to access your Google Drive files.
5. Google Colab may run very slowly. You may need to reduce the number of bootstrap samples. Execution of an entire grid of ARMA(p,q) models may not be viable.



# How to obtain the input data

1. Download or clone the [GitHub repository for the original StopptCOVID study](https://github.com/robert-koch-institut/StopptCOVID-Studie_Daten_Analyse_und_Ergebnisse)
2. The R language is required to run the StopptCOVID code requires R. If you do not have R installed already, you can download it, e.g., from [https://posit.co/download/rstudio-desktop/] (R and RStudio desktop recommended). R can also be installed via Anaconda.
3. Download the NPI data from infas:
   a. If you do not have an account for [https://www.healthcare-datenplattform.de], please register here: [https://www.healthcare-datenplattform.de/contact]
   b. Download county-level data for NPI subcategories from [https://www.healthcare-datenplattform.de/dataset/massnahmen_unterkategorien_kreise]
   c. Place these data into the subdirectory `Daten\infas` in your `StopptCOVID-Studie_Daten_Analyse_und_Ergebnisse` directory.
4. Modify the code to output the input data for the regression model:
   a. Open `StopptCOVID-Studie_Daten_Analyse_und_Ergebnisse\Skripte\Main_model`, e.g., in RStudio or in another editor.
   b. Insert this line at the end of the section Data preparation and before Main model (around line 33):
    `save(data_main, file = file.path(Daten, data_main.rData))`
5. Run the R code (e.g., open `StopptCOVID_main.R` in RStudio and go to Code–>Run Region -> Run All):
   a. To avoid conflicts, you may need to change line in StopptCOVID_main.R from
      `library(tidyverse)`
      to
      `library(conflicted)`
      `library(tidyverse)`
      `conflict_prefer(filter, dplyr)`
      `conflict_prefer(lag, dplyr)`
      `library(tidyverse)`
   b. You may get error messages about missing packages. If so install these from
 the console,
      `install.packages(“<package name>”)`
or using your package management system, e.g., Tools->Install Packages. Rerun the code after installing packages until it executes without interruption.
   c. During a test with RStudio 2023.12.1 Build 402, the following packages had to be installed:
insight, margins, parameters, patchwork, performance, plyr, prediction, conflicted, cowplot, multcomp, mvtnorm,sandwich, TH.data, commonmark, ggtext, gridtext, jpeg, markdown, png, corrplot, ggthemes, wesanderson, bayestestR, datawizard, dotwhisker, ggstance, gridExtra
   d. The R code will automatically download any other required data other than the infas data. This may require some time.
6. After successful execution, the code will have generated a file `Daten\data_main.rData`. Copy this file to the working directory for the Python code, and the Python code will be ready to run.

# Contact
[Bernhard Mueller](bernhard.mueller@monash.edu)
