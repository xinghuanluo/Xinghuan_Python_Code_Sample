# =============================================================================
# This file has mainly 3 parts 
# Part 0: Preparation
# Part 1: Summarized Graph and Cross-Validation Prediction
# Part 3: Prediction with One Variable
# =============================================================================


import os 
import requests
from zipfile import ZipFile
import pandas as pd
import numpy.linalg as la
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt

# =============================================================================
# Part 0: Preparation
# =============================================================================
zip_path = r"https://opportunityinsights.org/wp-content/uploads/2019/05/project4.zip"
edu_path = r"https://www.ers.usda.gov/webdocs/DataFiles/48747/Education.xls?v=9714.9"

def join_path(file_dire):
    file_path = os.path.join(os.getcwd(), file_dire)
    return file_path

# Download zip data and convert them to dataframe
# I learned the below funtion from this link, https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
def download_url(url, save_path, fname):
    r = requests.get(url)
    with open(join_path(fname), 'wb') as ofile: 
        ofile.write(r.content)
        
download_url(zip_path, os.getcwd(), "file.zip")

with ZipFile("file.zip", 'r') as zip: 
    zip.extractall()
    
train_path = join_path("project4\\atlas_training.dta")
test_path = join_path("project4\\atlas_test.dta")

# The variables I need for later analysis
var_list = ['geoid','pop', 'housing', 'kfr_pooled_p25','training',
            'P_7', 'P_12', 'P_24', 'P_27', 'P_30', 
            'P_31', 'P_41', 'P_47', 'P_54', 'P_80']

pd_train = pd.read_stata(train_path, columns=var_list)
pd_train_label = pd.read_stata(train_path, columns=var_list, iterator=True)

# Below code is to get the variable's label in Stata so that I could check when I need
train_all_label = pd_train_label.variable_labels()
unknown_var = [i for i in var_list if i.startswith("P")]
unknown_label = {key: train_all_label.get(key) for key in unknown_var}

pd_test = pd.read_stata(test_path, columns=['geoid','kfr_actual'])

# download data from USDA website and convert them to dataframe
download_url(edu_path, os.getcwd(), "edu.xls")
required_columns = ['FIPS Code', 
                    'State', 
                    'Percent of adults with less than a high school diploma, 2000', 
                    'Percent of adults with a high school diploma only, 2000', 
                    "Percent of adults completing some college or associate's degree, 2000", 
                    "Percent of adults with a bachelor's degree or higher, 2000"]
pd_edu = pd.read_excel("edu.xls", skiprows=[0, 1, 2, 3, 5], usecols=required_columns)
pd_edu_new_col_name = ["geoid", "State", "below high school", "only high shool", "only college", "above college"]
pd_edu.columns = pd_edu_new_col_name

# Check if the geoid uniquely identifies the observations
for assert_pd in [pd_edu, pd_train, pd_test]: 
    assert assert_pd['geoid'].is_unique

# Merging train data, test data and edu data together. Only keep the intersection included in all 3 data sets
pd_train = pd_train.merge(pd_edu, how='inner', left_on='geoid', right_on='geoid')
all_data = pd_train.merge(pd_test, how='inner', left_on='geoid', right_on='geoid')

# Clean the all_data
all_data['kfr_pooled_p25'].fillna(all_data['kfr_actual'], inplace=True)
all_data.rename(columns={'kfr_pooled_p25': 'social_mobi'}, inplace=True)
state_col = all_data['State']
all_data.drop(['kfr_actual','pop', 'housing', 'State'], axis=1, inplace=True)
all_data.insert(0, "state", state_col)
all_data['training'] = all_data['training'].astype(int)

all_data.dropna(inplace = True)
assert not all_data.isnull().values.any()

assert (all_data['training'].unique() == [1, 0]).all()

# Summarize data with Plots
all_data_state = all_data.groupby(['state']).mean()
social_mobi = all_data_state['social_mobi']


# =============================================================================
# Part 1: Summarized Graph and Cross-Validation Prediction
# =============================================================================
fig, ax = plt.subplots(figsize=(12,6))
n, bins, patches = ax.hist(social_mobi, bins=10, edgecolor='black', color = "red",rwidth=0.8)
n = (n.astype(int)).tolist()
ax.set_ylabel('Frequency')
ax.set_xlabel("Values of Social Mobility Score")
ax.set_title('Distribution and Summary Statistics of Social Mobility Score Across Years by States')
bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
bin_center_list = np.arange(min(bins)+bin_w/2, max(bins), bin_w).tolist()
for index, x in enumerate(bin_center_list): 
    plt.text(x, n[index]+0.1, n[index], fontsize=12)
# I learned how to plot this text box from here, https://matplotlib.org/3.3.3/gallery/recipes/placing_text_boxes.html#sphx-glr-gallery-recipes-placing-text-boxes-py 
average = social_mobi.mean()
median = np.median(social_mobi)
sigma = social_mobi.std()
min_value = social_mobi.min()
max_value = social_mobi.max()
number_obs = (social_mobi.count()).astype(int)
textstr = '\n'.join((
    r'$\mu=%.2f$' % (average, ),
    r'$\mathrm{median}=%.2f$' % (median, ),
    r'$\sigma=%.2f$' % (sigma, ), 
    r'$\min=%.2f$' % (min_value, ), 
    r'$\max=%.2f$' % (max_value, ), 
    r'$number=%.2f$' % (number_obs, )))
props = dict(boxstyle='square', facecolor='blue', alpha=0.5)
plt.text(0.1, 0.95, textstr, transform=ax.transAxes, fontsize=16, 
        verticalalignment='top', bbox=props)    
ax.set_xticks(bin_center_list)
ax.set_xlim(bins[0], bins[-1])
plt.savefig("Distribution_Statistics.png")
plt.show()


# Use Cross Validation with OLS to predict "social_mobility"
x_columns = all_data.columns[4:]

def split_set(x, col): 
    output = x.loc[:, col].to_numpy()
    return output

idpt_var = split_set(all_data, x_columns)
dpdt_var = split_set(all_data, 'social_mobi')

def get_ols_error(X_train,X_test,Y_train, Y_test): 
    """
    This function is to get ols prediction error
    """
    w_hat = la.inv(X_train.T@X_train)@X_train.T@Y_train
    y_hat = X_test@w_hat
    change_rate = abs(Y_test - y_hat / Y_test)
    avg_rate = change_rate.mean()
    return avg_rate

def cross_split_data(i, group_num, X, Y, all_index): 
    """
    This function splits the data into test and training set
    """
    if (i+1)*group_num not in all_index: 
        test_index = np.arange(i*group_num, all_index[-1])
    else: 
        test_index = np.arange(i*group_num, (i+1)*group_num)
    train_index = np.delete(all_index, test_index)
    train_x = X[train_index]
    train_y = Y[train_index]
    test_x = X[test_index]
    test_y = Y[test_index]
    return train_x, train_y, test_x, test_y
    
def get_total_grp(X, grp_vol): 
    """
    This function calculate the number of group by given group size. 
    """
    total_grp = divmod(X.shape[0], grp_vol)[0]
    return total_grp


def cross_val_error(X, Y, grp_vol): 
    """
    This function is combined with previous functions 
    and is to calculate the average changing rate of OLS with cross-validation
    """
    all_index = list(range(X.shape[0]))
    total_grp = get_total_grp(X, grp_vol)
    error_list = []
    for i in list(range((total_grp+1))): 
        train_x, train_y, test_x, test_y = cross_split_data(i, grp_vol, X, Y, all_index)
        change_rate = get_ols_error(train_x, test_x, train_y, test_y)
        error_list.append(change_rate)
    changing_rate = sum(error_list)/len(error_list)
    percent = "{:.2%}".format(changing_rate)
    return percent
    
changing_percent = cross_val_error(idpt_var, dpdt_var, 100)
print(f"\n Compared with the true data, the result of cross validation has changed {changing_percent}")

# =============================================================================
# Part 3: Prediction with One Variable
# =============================================================================
# I choosed the variable with the highest weight and do the OLS again 
all_data_state.reset_index(inplace=True)
half_index = int(len(all_data_state)/2)
train_state = all_data_state.loc[:half_index, :]
test_state = all_data_state.loc[half_index:, :]

def get_w_hat(train_set, test_set, x_var, y_var): 
    """
    This function get w hat.
    """
    train_x = split_set(train_set, x_var)
    train_y = split_set(train_set, y_var)
    test_x = split_set(test_set, x_var)
    test_y = split_set(test_set, y_var)  
    train_x = np.reshape(train_x, (train_x.shape[0], -1))
    w_hat = la.inv(train_x.T@train_x)@train_x.T@train_y
    return w_hat, test_x, test_y
        
w_state_hat, test_x, test_y = get_w_hat(train_state, test_state, x_columns, 'social_mobi')
w_state_hat_abs = list(enumerate(abs(w_state_hat)))

# I learned the below code from this link, https://www.tutorialspoint.com/get-first-element-with-maximum-value-in-list-of-tuples-in-python
max_index = max(w_state_hat_abs, key=itemgetter(1))[0]
w_optimal_hat, test_x, test_y = get_w_hat(train_state, test_state, x_columns[max_index], 'social_mobi')

# I learned the below graph from here. https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.colorbar.html
fig, ax = plt.subplots(figsize=(8, 4))
x_axis = np.linspace(0.1, 0.4)
line_graph = plt.plot(x_axis, x_axis*w_optimal_hat, label="OLS prediction")
scatter_graph = plt.scatter(test_x, test_y, c=test_y, label="Actual y")
plt.legend()
plt.xlabel("Fraction of Residents w/ a College Degree 20006 - 2010")
plt.ylabel("Mean Rank of the Child Future Income")
plt.colorbar()
plt.savefig("Significant_Variable_Prediction.png")
plt.show()

# Output the final dataset
all_data.to_csv('Output_Data.csv', index=False)
