import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("seaborn-darkgrid")
import datetime as dt
import statsmodels.api as sm
from scipy.stats import shapiro, f_oneway, levene, ttest_ind, chi2_contingency
import statsmodels.stats.api as sms
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.eval_measures import rmse, mse, meanabs
from sklearn.metrics import r2_score
import math

class print_format:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    @staticmethod
    def print_line(num_lines = 100):
        print(print_format.BOLD + print_format.GREEN + "-"*num_lines + print_format.END)
    

class BasicDataChecks:
    pd.set_option('expand_frame_repr', False)
    
    def __init__(self, data_loc):
        self.data = pd.read_csv(data_loc, index_col=0)
        self.index_check()
        self.rename_cols()
    
    def index_check(self):
        if self.data.index[0] == 1:
            self.data.index = self.data.index - 1
    
    def rename_cols(self):
        self.data.columns = self.data.columns.str.lower()
        self.data.columns = self.data.columns.str.replace(" ", "_")
    
    def check_null(self):
        return self.data.isnull().sum()
    
    def check_duplicates(self):
        if self.data.duplicated().sum() > 0:
            return self.data[self.data.duplicated(keep=False)]
        return self.data.duplicated().sum()
    
    @property
    def check(self):
        print_format.print_line()
        print(print_format.BOLD + "Data head: " + print_format.END)
        print(self.data.head())
        print_format.print_line()
        print(print_format.BOLD + "Data tail:  " + print_format.END)
        print(self.data.tail())
        print_format.print_line()
        print(print_format.BOLD + "Null Values: " + print_format.END)
        print(self.check_null())
        print_format.print_line()
        print(print_format.BOLD + "Duplicate Values: " + print_format.END)
        print(self.check_duplicates())
        print_format.print_line()
        print(print_format.BOLD + "Data Info: " + print_format.END)
        print(self.data.info())
        print_format.print_line()
        print(print_format.BOLD + "Data Description: " + print_format.END)
        print(self.data.describe())
        print_format.print_line()

        
class ExploratoryDataAnalysis:
    def __init__(self, data: pd.DataFrame, cat_vars: list):
        self.data = data
        self.cat_vars = cat_vars
       
    def frequency_plot(self, width = 20, height = 5, ncols = 4):
        nrows = math.ceil(len(self.cat_vars) / ncols)
        row_cur = 0
        ncols = ncols
        
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
        cols = self.cat_vars
        for num, col in enumerate(cols):
            plot = sns.countplot(data = self.data, x = col, ax = ax[row_cur][num % ncols])
            ax[row_cur][num % ncols].set_title(f"{col}", fontsize=14)
            ax[row_cur][num % ncols].set_xticklabels(ax[row_cur][num % ncols].get_xticklabels(), rotation=90, ha='right')
            if (num % ncols == ncols - 1):
                row_cur += 1
        plt.show()
        return None
    
    def violin_plot(self, width = 20, height = 5, ncols = 4):
        nrows = math.ceil((len(self.data.columns) - len(self.cat_vars)) / ncols)
        row_cur = 0
        ncols = ncols
        
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
        cols = self.data.columns[~self.data.columns.isin(pd.Index(self.cat_vars))].tolist()
        for num, col in enumerate(cols):
            sns.violinplot(data = self.data[col], ax=ax[row_cur][num % ncols], 
                           color = (np.random.random(), np.random.random(), np.random.random()), inner = 'box', grid=True)
                
            ax[row_cur][num % ncols].set_title(f"{col}", fontsize=14)
            ax[row_cur][num % ncols].get_children()[1].set_color('k')
            ax[row_cur][num % ncols].get_children()[1].set_lw(5)
            
            ax[row_cur][num % ncols].get_children()[2].set_color('w')
            ax[row_cur][num % ncols].get_children()[3].set_color('w')
            ax[row_cur][num % ncols].set_xticks([])
            ax[row_cur][num % ncols].axhline(self.data[col].mean(), color = 'pink', lw = 4)
            ax[row_cur][num % ncols].legend({f'Mean {round(self.data[col].mean(), 2)}':self.data[col].mean()})
            
            if (num % ncols == ncols - 1):
                row_cur += 1
                
        plt.show()
        
    def hist_plot(self, width = 20, height = 5, ncols = 4):
        nrows = int((len(self.data.columns) - len(self.cat_vars)) / 4)
        row_cur = 0

        ncols = ncols
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
        cols = self.data.columns[~self.data.columns.isin(pd.Index(self.cat_vars))].tolist()
        for num, col in enumerate(cols):
            sns.histplot(data = self.data[col], ax=ax[row_cur][num % ncols], kde=True)          
            if (num % ncols == ncols - 1):
                row_cur += 1

        plt.show()
        
    def box_plot(self, width = 20, height = 5, ncols = 4):
        nrows = int((len(self.data.columns) - len(self.cat_vars)) / 4)
        row_cur = 0

        ncols = ncols
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
        cols = self.data.columns[~self.data.columns.isin(pd.Index(self.cat_vars))].tolist()
        for num, col in enumerate(cols):
            sns.boxplot(data = self.data[[col]], ax=ax[row_cur][num % ncols])          
            if (num % ncols == ncols - 1):
                row_cur += 1

        plt.show()
    
    def univariate_analysis(self, num_lines = 150, width = 20, height = 5, ncols = 4, violin = True, hist = True, box = True, norm_check = True):
        if len(self.cat_vars)!=0:
            print(print_format.UNDERLINE + print_format.PURPLE + print_format.BOLD + "Count of Categorical Variables: \n" + print_format.END)
            self.frequency_plot(width = width, height = height, ncols = ncols)
        
        if violin:
            print(print_format.UNDERLINE + print_format.PURPLE + print_format.BOLD + "Violin Plot of Continuous Variables: \n" + print_format.END)
            self.violin_plot(width = width, height = height, ncols = ncols)
            
        if hist:
            print(print_format.UNDERLINE + print_format.PURPLE + print_format.BOLD + "\nDistribution of Continuous Variables: \n" + print_format.END)
            self.hist_plot(width = width, height = height, ncols = ncols)
            
        if box:
            print(print_format.UNDERLINE + print_format.PURPLE + print_format.BOLD + "\nOutliers for Continuous Variables: \n" + print_format.END)
            self.box_plot(width = width, height = height, ncols = ncols)
            
        if norm_check:
            cols = self.data.columns[~self.data.columns.isin(pd.Index(self.cat_vars))].tolist()
            data_norm_check = self.data[cols]
            hypo_test = HypothesisTesting(data_norm_check)
            for num, col in enumerate(cols):
                hypo_test.normality_check(data_norm_check[col], check_log_normal=False)
                
        
    def correlation(self, method='pearson'):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        fig.suptitle("Variable Correlation", fontsize=20)
        cols = self.data.select_dtypes([np.number]).columns
        corr = self.data[cols].corr(method = method).abs().unstack()
        sns.heatmap(data = self.data[cols].corr(method = method), annot=True, lw=0.2, cmap='Greens')
        
    @staticmethod
    def plot_heatmap(cross_table, fmt='g'):
        fig, ax = plt.subplots(figsize=(8, 5))
        heatmap = sns.heatmap(cross_table,
                    annot=True,
                    fmt=fmt,
                    cmap='rocket_r',
                    linewidths=.5,
                    ax=ax)
        heatmap.set_xticklabels(heatmap.get_xticklabels(), 
                                rotation=45, 
                                horizontalalignment='right', fontsize=14)
        heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14)
        
        plt.show();
        return ax
    
    def contigency_table(self, index, columns, normalize=False, margins=False, margins_name=None, values=None, aggfunc=None, plot=False, fmt='g'):
        cross_data = self.data
        if values:
            values = cross_data[values]
            
        cross_table = pd.crosstab(index = [cross_data[idx] for idx in index], columns = [cross_data[col] for col in columns], 
                          normalize=normalize, 
                          margins=margins, margins_name=margins_name, 
                          values=values, aggfunc=aggfunc)
        
        if plot:
            ExploratoryDataAnalysis.plot_heatmap(cross_table, fmt=fmt)
        else:
            return cross_table
        
    def pair_plot(self, hue = None, kind='scatter', palette='Set1', kde=True):
        g = sns.pairplot(data=self.data, hue=hue, kind=kind, palette=palette)
        if kde:
            g.map_lower(sns.kdeplot, levels=4, color=".1")
    
    def bivariate_analysis(self, corr=False, pairplot=False, contigency_table = False, catplot=False):
        if corr:
            # Correlation
            print(print_format.UNDERLINE + print_format.PURPLE + print_format.BOLD + "Pearson's Correlation between Continuous Variables: \n" + print_format.END)
            self.correlation('pearson')
            plt.show()

            # Correlation
            print(print_format.UNDERLINE + print_format.PURPLE + print_format.BOLD + "\nSpearman's Correlation between Continuous Variables: \n" + print_format.END)
            self.correlation('spearman')
            plt.show()
        
        if pairplot:
            # Pairplot
            print(print_format.UNDERLINE + print_format.PURPLE + print_format.BOLD + "Pairplot with Research Experience: \n" + print_format.END)
            self.pair_plot(hue = 'research')
            plt.show()

            # Pairplot
            print(print_format.UNDERLINE + print_format.PURPLE + print_format.BOLD + "\nPairplot with University Rating: \n" + print_format.END)
            self.pair_plot(hue = 'university_rating', kde=False)
            plt.show()
            
        if contigency_table:
            print(print_format.UNDERLINE + print_format.PURPLE + print_format.BOLD + "University Rating and Reasearch Experience: \n" + print_format.END)
            self.contigency_table(index=["research"], columns = ["university_rating"], normalize='index', plot=True, fmt='.2%')
            plt.show()
        
        if catplot:
            print(print_format.UNDERLINE + print_format.PURPLE + print_format.BOLD + "Chances of Admit Based on Research Experience for Different University Rating: \n" + print_format.END)
            sns.catplot(data = self.data, x='research', y = 'chance_of_admit_', col='university_rating', kind='bar', sharex=False, height=4, aspect=0.6, palette='hot', ci=None)
            plt.show()
            
            print(print_format.UNDERLINE + print_format.PURPLE + print_format.BOLD + "\nCGPA & the Relationship b/w Research Experience + Different University Rating: \n" + print_format.END)
            sns.catplot(data = self.data, x='research', y = 'cgpa', col='university_rating', kind='bar', sharex=False, height=4, aspect=0.6, palette='hot', ci=None)
            plt.show()
            
            print(print_format.UNDERLINE + print_format.PURPLE + print_format.BOLD + "\nSOP Strength & the Relationship b/w Research Experience + Different University Rating: \n" + print_format.END)
            sns.catplot(data = self.data, x='research', y = 'sop', col='university_rating', kind='bar', sharex=False, height=4, aspect=0.6, palette='hot', ci=None)
            plt.show()
            
            print(print_format.UNDERLINE + print_format.PURPLE + print_format.BOLD + "\nLOR Strength & the Relationship b/w Research Experience + Different University Rating: \n" + print_format.END)
            sns.catplot(data = self.data, x='research', y = 'lor_', col='university_rating', kind='bar', sharex=False, height=4, aspect=0.6, palette='hot', ci=None)
            plt.show()
            
            print(print_format.UNDERLINE + print_format.PURPLE + print_format.BOLD + "\nGRE Score & the Relationship b/w Research Experience + Different University Rating: \n" + print_format.END)
            sns.catplot(data = self.data, x='research', y = 'gre_score', col='university_rating', kind='bar', sharex=False, height=4, aspect=0.6, palette='hot', ci=None)
            plt.show()
            
            print(print_format.UNDERLINE + print_format.PURPLE + print_format.BOLD + "\nTOEFL Score & the Relationship b/w Research Experience + Different University Rating: \n" + print_format.END)
            sns.catplot(data = self.data, x='research', y = 'toefl_score', col='university_rating', kind='bar', sharex=False, height=4, aspect=0.6, palette='hot', ci=None)
            plt.show()
        
        
class HypothesisTesting:
    def __init__(self, data):
        self.data = data
    
    def _plot_qq_hist(self, variable, alpha):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Normality Check for {variable.name}")
        sm.qqplot(np.array(variable), line='s', ax = ax[0])
        ax[0].set_title("Q-Q Plot")
        sns.histplot(variable, kde=True, ax = ax[1])
        ax[1].set_title("Distribution Plot")
        plt.show()
    
    def _shapiro_wilk(self, variable, alpha):
        pval = shapiro(variable).pvalue
        if pval > alpha:
            s1 = f"Shapiro-Wilk p-val: {round(pval, 2)} | alpha: {alpha}\nWe do not have sufficient evidence to say that {variable.name} doesn't come \nfrom a normal distribution."
            print_format.print_line(85)
            print(print_format.GREEN + print_format.BOLD + s1 + print_format.END)
            print_format.print_line(85)
        else:
            s2 = f"Shapiro-Wilk p-val: {round(pval, 2)} | alpha: {alpha}\nWe have sufficient evidence to say that {variable.name} doesn't come \nfrom a normal distribution."
            print_format.print_line(85)
            print(print_format.RED + print_format.BOLD + s2 + print_format.END)
            print_format.print_line(85)
            
    def residuals_plot(self, x, y):
        print_format.print_line(85)
        print(print_format.PURPLE + print_format.BOLD + "Residuals Check" + print_format.END)
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        sns.scatterplot(self.data[x], y = self.data[y], ax = ax, data = self.data)
        ax.text(0.10, 0.92, f"Mean of {x}: {round(self.data[x].mean(), 2)}")
        ax.set_title("Residuals Plot")
        plt.show()
        
    def normality_check(self, variable, alpha = 0.05, check_log_normal = True):
        print_format.print_line(85)
        print(print_format.PURPLE + print_format.BOLD + f"Normality Check for: {variable.name}\n" + print_format.END)
        self._plot_qq_hist(variable, alpha)
        self._shapiro_wilk(variable, alpha)
        
        if check_log_normal:
            print(print_format.CYAN + print_format.BOLD + "Checking for Normality after Log-transformation:\n" + print_format.END)
            variable = np.log(variable)
            self._plot_qq_hist(variable, alpha)
            self._shapiro_wilk(variable, alpha)
            
    def homoscedasticity_check(self, y, x, idx=None, split=None, drop=None, alternative='increasing', store=False, alpha = 0.05):
        fval, pval, ordering = sms.het_goldfeldquandt(y = y, x = x, idx=idx, split=split, drop=drop, alternative=alternative, store=store)
        print_format.print_line(85)
        print(print_format.BLUE + print_format.BOLD + "Goldfeld-Quandt Homoskedasticity Test" + print_format.END)
        print(print_format.PURPLE+ print_format.BOLD + f"Alpha: {alpha} | pvalue: {round(pval, 2)}" + print_format.END)
        print(print_format.CYAN+ print_format.BOLD + "Null: Error terms are homoscedastic | Alternative: Error terms are heteroscedastic." + print_format.END)
        
        if pval > alpha:
            print(print_format.BOLD + print_format.RED + "We do not have sufficient evidence to reject the null hypothesis." + print_format.END)
        else:
            print(print_format.BOLD + print_format.GREEN + "We have sufficient evidence to reject the null hypothesis." + print_format.END)
        print_format.print_line(85)
        
    def variance_check(self, *args, center='median', proportiontocut=0.05):
        print_format.print_line(85)
        print(print_format.BLUE + print_format.BOLD + "Levene's Test For Variance" + print_format.END)
        var = [arg.name for arg in args]
        print(print_format.CYAN + print_format.BOLD + f"Variables: {var}" + print_format.END)
        
        pval = levene(*args, center='median', proportiontocut=proportiontocut).pvalue
        print(print_format.PURPLE+ print_format.BOLD + f"Proportion Cut: {proportiontocut} | pvalue: {round(pval, 2)}" + print_format.END)
        
        if pval > proportiontocut:
            print_format.print_line(85)
            s1 = "We do not have sufficient evidence to say that the variables don't have equal variance."
            print(print_format.GREEN + print_format.BOLD + s1 + print_format.END)
            print_format.print_line(85)
        else:
            print_format.print_line(85)
            s1 = "We have sufficient evidence to say that the variables don't have equal variance."
            print(print_format.RED + print_format.BOLD + s1 + print_format.END)
            print_format.print_line(85)
            
    def ind_ttest(self, a, b, axis=0, equal_var=True, nan_policy='propagate', permutations=None, random_state=None, alternative='two-sided', trim=0, alpha = 0.05, ret = False):
        stat, pval = ttest_ind(a, b, axis=0, equal_var=True, nan_policy='propagate', permutations=None, random_state=None, alternative='two-sided', trim=0)
        
        print_format.print_line(65)
        print(print_format.BOLD + print_format.CYAN + f"t-test | {a.name, b.name}" + print_format.END)
        print(print_format.RED + f"alpha: {alpha}" + print_format.END + " | " + print_format.BLUE + f"pval: {round(pval, 2)}" + print_format.END)
        
        if pval > alpha:
            print(print_format.BOLD + print_format.RED + "We do not have sufficient evidence to reject the null hypothesis." + print_format.END)
        else:
            print(print_format.BOLD + print_format.GREEN + "We have sufficient evidence to reject the null hypothesis." + print_format.END)
            
        print_format.print_line(65)
        if ret:
            return stat, pval
    
    def one_way_anova(self, *args, axis=0, alpha = 0.05, ret = False):
        stat, pval = f_oneway(*args, axis=0)
        
        print_format.print_line(65)
        print(print_format.BOLD + print_format.CYAN + f"one-way ANOVA" + print_format.END)
        print(print_format.RED + f"alpha: {alpha}" + print_format.END + " | " + print_format.BLUE + f"pval: {round(pval, 2)}" + print_format.END)
        
        if pval > alpha:
            print(print_format.BOLD + print_format.RED + "We do not have sufficient evidence to reject the null hypothesis." + print_format.END)
        else:
            print(print_format.BOLD + print_format.GREEN + "We have sufficient evidence to reject the null hypothesis." + print_format.END)
            
        print_format.print_line(65)
        if ret:
            return stat, pval
    
    def chi2(self, observed, correction=True, lambda_=None, alpha = 0.05, ret = False):
        stat, pval, dof, expected = chi2_contingency(observed, correction=True, lambda_=None)
        
        print_format.print_line(65)
        print(print_format.BOLD + print_format.CYAN + f"Chi-squared test" + print_format.END)
        print(print_format.RED + f"alpha: {alpha}" + print_format.END + " | " + print_format.BLUE + f"pval: {round(pval, 3)}" + print_format.END)
        
        if pval > alpha:
            print(print_format.BOLD + print_format.RED + "We do not have sufficient evidence to reject the null hypothesis." + print_format.END)
        else:
            print(print_format.BOLD + print_format.GREEN + "We have sufficient evidence to reject the null hypothesis." + print_format.END)
            
        print_format.print_line(65)
        if ret:
            return stat, pval
        
        
class LinearRegression:
    def __init__(self, data, endog_var, scale = True, scaling_method = MinMaxScaler):
        self.data = data
        self._endog_var = endog_var
        
        self.scale_data = scale
        self.scaling_method = scaling_method
                
        self.endog_var
        self.exog_vars
        
        self.model = None
        self.scaler_exog = None
        self.scaler_endog = None
        
    @property
    def endog_var(self):
        self._endog = self.data[[self._endog_var]]
        return self._endog
    
    @property
    def exog_vars(self):
        self._exog = self.data[self.data.columns[~self.data.columns.isin(self.endog_var.columns)].tolist()]
        return self._exog
    
    @property
    def _data(self):
        return self.data
    
    def one_hot_encoding(self, var_name):
        encoded_col = pd.get_dummies(self.data[var_name], prefix=var_name)
        self.data = pd.concat([self.data, encoded_col], axis = 1, )
        self.data.drop(columns = var_name, inplace=True)
    
    def split_train_test(self, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None, ret = False):
        self._exog_train, self._exog_test, self._endog_train, self._endog_test = train_test_split(self._exog, self._endog, 
                                                                                                  test_size=test_size, train_size=train_size, random_state=random_state, 
                                                                                                  shuffle=shuffle, stratify=stratify)
        if ret:
            return self._exog_train, self._exog_test, self._endog_train, self._endog_test
        
    def _scaler(self, train_data):
        self.scaler = self.scaling_method().fit(train_data)
        return self.scaler
    
    def multicollinearity_check(self, threshold = 5, remove_multi_col = True):
        if self.scale_data:
            scaled_train_dataset = self._scaler(self._exog_train).transform(self._exog_train)    
        else:
            scaled_train_dataset = self._exog_train.values
                
        variables = scaled_train_dataset
        vif = pd.DataFrame()
        vif["VIF"] = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
        vif["Features"] = self._exog_train.columns
        
        if remove_multi_col:
            vif['status'] = "Available"
            filter_ = vif['VIF'] > threshold
            cols_to_drop = vif[filter_]["Features"].to_list()
            vif.loc[filter_, 'status'] = "Dropped"
            
            self._exog_train.drop(columns = cols_to_drop, inplace=True)
            self._exog_test.drop(columns = cols_to_drop, inplace=True)
            
            print(print_format.BOLD + "Columns Dropped: " + str(cols_to_drop) + print_format.END)
        return vif
    
    def fit_ols_regression(self, add_constant = True, col_to_drop = None):
        self.add_constant = add_constant
        exog_train = self._exog_train
        
        if col_to_drop:
            exog_train.drop(columns = col_to_drop, inplace=True)
            self._exog_test.drop(columns = col_to_drop, inplace=True)
        
        if self.scale_data:
            self.scaler_exog = self._scaler(exog_train)
            exog_train = pd.DataFrame(data = self.scaler_exog.transform(exog_train), columns=exog_train.columns, index = exog_train.index)
            
        if self.add_constant:
            exog_train = sm.add_constant(exog_train)
            
        self.model = sm.OLS(endog=self._endog_train, exog=exog_train).fit()
        return self.model
    
    @property
    def model_summary(self):
        if self.model:
            return self.model.summary()
        raise NameError("Please train the model first.")
        
    def predict(self, exog_vars, single=True):
        if self.model:
            if self.scale_data:
                exog_vars_scaled = pd.DataFrame(self.scaler_exog.transform(exog_vars), columns = exog_vars.columns, index = exog_vars.index)
            else:
                exog_vars_scaled = exog_vars
            
            if self.add_constant:
                if single:
                    exog_vars_scaled = np.insert(exog_vars_scaled.values, 0, 1).reshape(1, -1)
                else:
                    exog_vars_scaled = sm.add_constant(exog_vars_scaled)
                    
            return pd.DataFrame(self.model.predict(exog_vars_scaled), columns=[f"{self._endog_var}_predicted"], index = exog_vars_scaled.index)
        raise NameError("Please train the model first.")
        
    @property
    def residual_analysis(self):
        if self.model:
            pred_values = self.predict(self._exog_train, single = False).squeeze()
            residuals = self._endog_train.squeeze() - pred_values
            resid_data = pd.DataFrame({
                "Actuals": self._endog_train.squeeze(),
                "Predicted": pred_values,
                "Residuals": residuals
            })
            
            hypo_test = HypothesisTesting(resid_data)
            hypo_test.residuals_plot('Residuals', 'Predicted')
            hypo_test.normality_check(resid_data['Residuals'], check_log_normal=False)
            hypo_test.homoscedasticity_check(resid_data['Residuals'], self._exog_train)
            
    def r2_score_adj(self, actual_values, predicted_values):
        r2_score_sklearn = r2_score(actual_values, predicted_values)
        
        n = len(actual_values)
        p = self.model.params.size
        
        r2_score_adjusted = 1-(1-r2_score_sklearn)*(n-1)/(n-p-1)

        return r2_score_adjusted
    
    @property
    def performance_analysis(self):
        performance = pd.DataFrame(columns = ["Train", "Test"], index = ["MAE", "MSE", "RMSE", "R^2 scikit-learn", "Adj. R^2 scikit-learn"])
        train_prediction = self.predict(self._exog_train, single = False).squeeze()
        test_prediction = self.predict(self._exog_test, single = False).squeeze()
        
        performance.loc["MAE"] = [round(meanabs(self._endog_train.squeeze(), train_prediction), 3), 
                                  round(meanabs(self._endog_test.squeeze(), test_prediction), 3)]
        performance.loc["MSE"] = [round(mse(self._endog_train.squeeze(), train_prediction), 3), 
                                  round(mse(self._endog_test.squeeze(), test_prediction), 3)]
        performance.loc["RMSE"] = [round(rmse(self._endog_train.squeeze(), train_prediction), 3), 
                                  round(rmse(self._endog_test.squeeze(), test_prediction), 3)]
        performance.loc["R^2 scikit-learn"] = [round(r2_score(self._endog_train.squeeze(), train_prediction),4), 
                                               round(r2_score(self._endog_test.squeeze(), test_prediction), 4)]
        performance.loc["Adj. R^2 scikit-learn"] = [round(self.r2_score_adj(self._endog_train.squeeze(), train_prediction), 4), 
                                                    round(self.r2_score_adj(self._endog_test.squeeze(), test_prediction), 4)]
        return performance
    

def lr_models(data, endog_var, scale = True, scaling_method = StandardScaler, test_size=0.2, random_state = 42, threshold = 4.5, remove_multi_col = False, add_constant=True, col_to_drop = None, residual_analysis = True):
    reg_model = LinearRegression(data = data, endog_var = endog_var, scale = scale, scaling_method = scaling_method)
    reg_model.split_train_test(test_size=test_size, random_state = random_state)
    reg_model.multicollinearity_check(threshold = threshold, remove_multi_col = remove_multi_col)
    trained_model = reg_model.fit_ols_regression(add_constant=add_constant, col_to_drop = col_to_drop)
    if residual_analysis:
        reg_model.residual_analysis

    performance = reg_model.performance_analysis
    return reg_model, trained_model, performance