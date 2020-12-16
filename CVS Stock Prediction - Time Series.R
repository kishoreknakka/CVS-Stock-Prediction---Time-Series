#Project - CVS Stock prediction
#KISHORE KUMAR NAKKA (ND5674)

#########################     Step 1    #########################
#Step 1 - Goal
# CVS Stock prediction
library(forecast)
library(zoo)

#########################     Step 2    #########################
#Step 2 - Data
setwd("~/Desktop/TS/Project")
CVS.data <- read.csv("CVS.csv")
CVS.ts <- ts(CVS.data$Close, start = c(2016, 1), end = c(2019, 12), freq = 12)
CVS.ts

#########################     Step 3    #########################
#Step 3 - Explore and Visualizing

#ARIMA - to predict
CVS.ar1 <- Arima(CVS.ts, order = c(1,0,0))
summary(CVS.ar1)

CVS.ar2 <- Arima(CVS.ts, order = c(2,0,0))
summary(CVS.ar2)

diff.price <- diff(CVS.ts, lag = 1)
diff.price

Acf(diff.price, lag.max = 12, main = "Autocorrelation for the first difference (Lag 1) of CVS Close Prices from 2016")
#This indicates it is not a random walk

#VISUALIZING DATA
plot(CVS.ts, xlab = "Time", ylab = "Price", ylim = c(50, 110), main = "CVS stock", col = "blue")
CVS.stl <- stl(CVS.ts, s.window = "periodic")
autoplot(CVS.stl, main = "Stock Time Series Components")

# identifying possible time series components. 
autocor <- Acf(CVS.ts, lag.max = 12, main = "Autocorrelation for CVS Stock Price")

# Display autocorrelation coefficients for various lags.
Lag <- round(autocor$lag, 0)
ACF <- round(autocor$acf, 3)
data.frame(Lag, ACF)

#########################     Step 4    #########################
#Step 4 - Pre-processing Data (Nothing is required)
#########################     Step 5    #########################
#Step 5 - Partition

# partitioning the data to train and valid
nValid <- 12
nTrain <- length(CVS.ts) - nValid
train.ts <- window(CVS.ts, start = c(2016, 1), end = c(2016, nTrain))
train.ts
valid.ts <- window(CVS.ts, start = c(2016, nTrain + 1), end = c(2016, nTrain + nValid))
valid.ts

#########################     Step 6    #########################
#Step 6 - Forecasting Methods

#naive #
round(accuracy((naive(CVS.ts))$fitted, CVS.ts), 3)
# CVS.naive.pred <- naive(train.ts, h = nValid)
# round(accuracy(CVS.naive.pred$mean, valid.ts), 3)

#Seasonal naive
round(accuracy((snaive(CVS.ts))$fitted, CVS.ts), 3)
# CVS.snaive.pred <- snaive(train.ts, h = nValid)
# round(accuracy(CVS.snaive.pred$mean, valid.ts), 3)
#CVS.snaive <- snaive(CVS.ts, h=12, level = 0)
#round(accuracy(CVS.snaive$fitted, CVS.ts), 3)

#Moving average trailing
ma.trail_2 <- round(rollmean(CVS.ts, k = 2, align = "right"), 2)
ma.trail_4 <- round(rollmean(CVS.ts, k = 4, align = "right"), 2)
ma.trail_5 <- round(rollmean(CVS.ts, k = 5, align = "right"), 2)
ma.trail_6 <- round(rollmean(CVS.ts, k = 6, align = "right"), 2)
ma.trail_8 <- round(rollmean(CVS.ts, k = 8, align = "right"), 2)
ma.trail_12 <- round(rollmean(CVS.ts, k = 12, align = "right"), 2)
#ma.trail_12.pred <- forecast(ma.trail_12, h=12, level = 0)
#ma.trail_12.pred
# Combine CVS.ts and ma.trailing in one data table.
ma.trail_2 <- c(rep(NA, length(CVS.ts) - length(ma.trail_2)), ma.trail_2)
ma.trail_4 <- c(rep(NA, length(CVS.ts) - length(ma.trail_4)), ma.trail_4)
ma.trail_5 <- c(rep(NA, length(CVS.ts) - length(ma.trail_5)), ma.trail_5)
ma.trail_6 <- c(rep(NA, length(CVS.ts) - length(ma.trail_6)), ma.trail_6)
ma.trail_8 <- c(rep(NA, length(CVS.ts) - length(ma.trail_8)), ma.trail_8)
ma.trail_12 <- c(rep(NA, length(CVS.ts) - length(ma.trail_12)), ma.trail_12)
ma_trailing_tab <- cbind(CVS.ts, ma.trail_2, ma.trail_4, ma.trail_5, ma.trail_6, ma.trail_8, ma.trail_12)

#ma_trailing_tab

# identify common accuracy measures
round(accuracy(ma.trail_2, CVS.ts), 3) #Best
round(accuracy(ma.trail_4, CVS.ts), 3)
round(accuracy(ma.trail_5, CVS.ts), 3)
round(accuracy(ma.trail_6, CVS.ts), 3)
round(accuracy(ma.trail_8, CVS.ts), 3)
round(accuracy(ma.trail_12, CVS.ts), 3)

# Create Holt-Winter's exponential smoothing (HW)
CVS.ZZZ <- ets(train.ts, model = "ZZZ")
CVS.ZZZ
CVS.ZZZ.pred <- forecast(CVS.ZZZ, h = 12, level = 0)
round(accuracy(CVS.ZZZ.pred, valid.ts), 3)


#Regression based models

#  i. Regression model with linear trend
train.lin <- tslm(train.ts ~ trend)
summary(train.lin) #best2
train.lin.pred <- forecast(train.lin, h = nValid, level = 0)

# ii. Regression model with quadratic trend
train.quad <- tslm(train.ts ~ trend + I(trend^2))
summary(train.quad)
train.quad.pred <- forecast(train.quad, h = nValid, level = 0)

# iii. Regression model with seasonality
train.season <- tslm(train.ts ~ season)
summary(train.season)
train.season.pred <- forecast(train.season, h = nValid, level = 0)

# iv. Regression model with linear trend and seasonality
train.linear.trend.season <- tslm(train.ts ~ trend + season)
summary(train.linear.trend.season) #Best1
train.linear.trend.season.pred <- forecast(train.linear.trend.season, h = nValid, level = 0)

# v. Regression model with quadratic trend and seasonality.
train.trend.season <- tslm(train.ts ~ trend + I(trend^2) + season)
summary(train.trend.season)
train.trend.season.pred <- forecast(train.trend.season, h = nValid, level = 0)


# Comparing for the most accurate regression model for forecasting.
round(accuracy(train.lin.pred, CVS.ts), 3) #best2
round(accuracy(train.quad.pred, CVS.ts), 3)
round(accuracy(train.season.pred, CVS.ts), 3)
round(accuracy(train.linear.trend.season.pred, CVS.ts),3) #Best1
round(accuracy(train.trend.season.pred, CVS.ts),3)

# Auto Arima model
train.auto.arima <- auto.arima(train.ts)
summary(train.auto.arima)
train.auto.arima.pred <- forecast(train.auto.arima, h = 12, level = 0)
round(accuracy(train.auto.arima.pred, valid.ts), 3)

#########################     Step 7    #########################
#Step 7 - Evaluating and comparing
round(accuracy(train.lin.pred$mean, CVS.ts), 3) ##Reg model with linear trend
round(accuracy(train.linear.trend.season.pred$mean, CVS.ts),3) #Reg model with linear trend and seasonality

#########################     Step 8    #########################
#Step 8 - Implementing
#Regression model with linear trend and seasonality
CVS <- tslm(CVS.ts ~ trend + season)
CVS.pred <- forecast(CVS, h = 12, level = c(0, 95))
CVS.pred
#This is perfect

