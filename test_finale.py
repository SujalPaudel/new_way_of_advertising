import pandas as pd
from numpy import *
#import numpy as np
num_websites = 5
num_users = 3
num_features = 5

user_ratings = pd.read_csv('/home/velar/Documents/python/user_rating.csv')

print (user_ratings)

user_rating_conversion = np.loadtxt(open("/home/velar/Documents/python/user_rating.csv", "rb"), delimiter=",", skiprows = 1, usecols = (1, 2, 3))

user_did_rate = (user_rating_conversion !=0) *1

def normalized_rating(user_rating_conversion, user_did_rate):
    
    ratings_mean = np.zeros(shape = (num_websites, 1))
    ratings_norm = np.zeros(shape = (user_rating_conversion.shape))
    
    for i in range(num_websites):
        idx = np.where(user_did_rate[i] == 1)[0]
        ratings_mean[i] = np.mean(user_rating_conversion[i, idx])
        ratings_norm[i, idx] = user_rating_conversion[i, idx] - ratings_mean[i]
    
    return ratings_mean, ratings_norm
        
    
    
row_ko_mean_rate, normalized_rating = normalized_rating(user_rating_conversion, user_did_rate)

#this includes the user_features  
user_features = pd.read_csv('/home/velar/Documents/python/user_features.csv')
print (user_features)

user_features_conversion = np.loadtxt(open("/home/velar/Documents/python/user_features.csv", "rb"), delimiter=",", skiprows = 1, usecols = (1, 2, 3, 4, 5))
print(user_features_conversion)

#this includes the websites features
websites_features =  pd.read_csv('/home/velar/Documents/python/website_features.csv')
print (websites_features)

websites_features_conversion = np.loadtxt(open("/home/velar/Documents/python/website_features.csv", "rb"), delimiter=",", skiprows = 1, usecols = (1, 2, 3, 4, 5))
print(websites_features_conversion)

print(user_features_conversion)

user_rated_features = (user_features_conversion != 0) *1
print(user_rated_features)

print(websites_features_conversion) 

websites_features_mean = np.mean(websites_features_conversion, axis = 1)
newshape = (num_websites, 1)
mean_de_website_features = np.reshape(websites_features_mean, newshape)
print(mean_de_website_features)

normalized_websites_features = websites_features_conversion - mean_de_website_features

print(normalized_websites_features) # Website Features: (1 Business 2 Mobile/Laptop 3 Consultancy 4 Traveling 5 Movies) 

user_features_conversion # User Features: (1 Business 2 Mobile/Laptop 3 Consultancy 4 Traveling 5 Movies)

user_features_conversion_mean = np.mean(user_features_conversion, axis = 1)
shape = (num_users, 1)
reshapedV2 = user_features_conversion_mean.reshape(shape)
print(reshapedV2)

fine_reshapedV2 = reshapedV2 *  user_rated_features
normalized_user_features = user_features_conversion - fine_reshapedV2
print(normalized_user_features)

#st = user_features_conversion - stpt
#print (st)
#stpt = reshapedV2 *  user_rated_features
#print (stpt)

this_is_the_final_one = np.r_[normalized_websites_features.T.flatten(), normalized_user_features.T.flatten()]

print(this_is_the_final_one)

def reconstruction(this_is_the_final_one, num_users, num_websites, num_features):
    first_25 = this_is_the_final_one[:num_websites * num_features] 
    websites_features = first_25.reshape(num_features, num_websites).transpose()
    last_15 = this_is_the_final_one[num_websites * num_features:]
    user_features = last_15.reshape(num_features, num_users).transpose()
    return websites_features, user_features #these are normalized web and user features

    
user_features_conversion

print(user_did_rate)


#So, how do we actually come up with the line that best fits???
#That is the reason we use the gradient descent

# Gradient Descent:
#--> The minimum slope enables us to find the global minimum cost(the sum of squared error), 
#--> we go on decreasing descenting the slope until we find the lowest cost function(sum of squared error)
#--> The lowest sum of the squared error means the line of the best fit

def calculate_gradient(this_is_the_final_one, user_rating_conversion, user_did_rate, num_users, num_websites, num_features):
    websites_features, user_features= reconstruction(this_is_the_final_one, num_users, num_websites, num_features)
    
    #websites_features_grad = 5 * 5
    #user_features_grad = 
    
    #the dot product of websites_features and the user_features gives the user affinity:(lagav) for each of the genres
    difference = websites_features.dot(user_features.T) * user_did_rate - user_rating_conversion
    
    #we are calculating the partial_fraction(gradient(slope)) with respect to the cost function.
    
    websites_features_grad = difference.dot(user_features) 
    user_features_grad = difference.T.dot( websites_features ) 
    
    #wrap gradients back to the list
    return np.r_[websites_features_grad.T.flatten(), user_features_grad.T.flatten()]

    

def calculate_sum_of_squared_error(this_is_the_final_one, user_rating_conversion, user_did_rate, num_users, num_websites, num_features):
    websites_features, user_features = reconstruction(this_is_the_final_one, num_users, num_websites, num_features)
        
    #cost is the sum of squared distances between the axis and the line.
    #as shown, axis, or the plotted points are (X.dot(theta.t) * did_rate),
    #While the straight line is given by the ratings
    cost =  sum( (websites_features.dot(user_features.T) * user_did_rate - user_rating_conversion) ** 2 )/ 2
    
    # ** means an element wise power
    return cost



#a package used for advanced optimization, minimize cost function in this case
from scipy import optimize

#this makes website_predictions, from which we can make movie_recommendation
#optimize to the minimum value of the cost function
minimized_cost_and_optimal_params = optimize.fmin_cg(calculate_sum_of_squared_error, 
                                                     fprime=calculate_gradient, 
                                                      x0=this_is_the_final_one, 
                                                      args = (user_rating_conversion, user_did_rate,
                                                      num_users, num_websites, num_features),
                                                      maxiter = 100, disp = True, full_output = True)


cost, optimal_website_features_and_user_preferences = minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]


websites_features, user_preferences = reconstruction(optimal_website_features_and_user_preferences, num_users, num_websites, num_features )


all_predictions = websites_features.dot(user_preferences.T) #regular regression(y) = x.theta


print(all_predictions)

predictions = all_predictions[:, 0:3] + row_ko_mean_rate

print(predictions)