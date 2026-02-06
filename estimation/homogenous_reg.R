# Load libraries
library(dplyr)
library(readr)
library(fixest)  # for high-dimensional fixed effects regression

# Load the data
df <- read_csv("processed_data/blp_inst_US.csv")

# Make prices right scale
df <- df %>% mutate(
  distw = log(distw),
  prices = prices * 10
)



# Filter as needed

df <- df %>% filter(
  shares > 0.00001,
  prices < 10,  #state %in% states
  old_model == 0,
  next_model == 0
)


# Try un-logged vars
#df <- df %>% mutate(
#  hpwt = exp(hpwt),
#  mpg = exp(mpg),
#  size = exp(size)
#)

# Step 1: Compute outside share (1 - sum of inside shares per market)
df <- df %>%
  group_by(market_ids) %>%
  mutate(outside_share = 1 - sum(shares)) %>%
  ungroup()

# Step 2: Compute log-difference (delta)
df <- df %>%
  mutate(delta = log(shares) - log(outside_share))

# Step 3: Run the homogeneous logit regression
# We'll use prices, hpwt (horsepower per weight), and size as X variables
# Fixed effects on market (and optionally state or year if you want)

model <- feols(
  delta ~ prices + hpwt + size + mpg + factor(vehicle_type) +
    factor(engine_type) | 
  factor(market_ids) + factor(firm_ids),  # add `+ year` or `+ state` if desired
  data = df
)

# Step 4: Output regression results # OLS HERE
summary(model)


### Do 2SLS regression
#instrument_vars <- c(paste0("demand_instruments", 0:2))
#instrument_vars2 <- c(paste0("demand_instruments", 0:5))

rhs <- paste(c("rer_lag1","hpwt", "size", "mpg", 
               "vehicle_type", "engine_type", "firm_ids", 
               "factor(market_ids)"), 
             collapse = " + ")
first_stage_formula <- as.formula(paste("prices ~", rhs))

first_stage_model <- lm(first_stage_formula, data = df)
df$fitted_prices <- fitted(first_stage_model)

summary(first_stage_model) #FIRST STAGE HERE

second_stage_formula <- as.formula(
  "delta ~ hpwt + size + mpg + 
  vehicle_type + engine_type + firm_ids +
  factor(market_ids) + fitted_prices"
)

second_stage_model <- lm(second_stage_formula, data = df)
summary(second_stage_model) ### IV REGRESSION

### Get own-price elasticities

# Extract the price coefficient from second-stage model
beta_p <- coef(second_stage_model)["fitted_prices"]
beta_p
# Compute elasticity for each observation
df$own_price_elasticity <- beta_p * df$prices * (1 - df$shares)

# Compute mean own-price elasticity
mean_own_elasticity <- mean(df$own_price_elasticity, na.rm = TRUE)
print(mean_own_elasticity)










