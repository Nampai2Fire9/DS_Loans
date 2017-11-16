#  EDA on Prosper Loan API data 


library(caret)
library(doSNOW)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(gridExtra)    #Grids on graphs 
library(ggcorrplot)   # Correlation matrix 
library(polycor)      #Pont Biserial Correlations 
library(ltm) 
library(grid)
library(gridExtra)    #allows for grids on graphs 
library(scales)       #formats of grpahs
library(stringr)
library(dplyr)
library(plyr)         #summary statistical tables 
library(ROCR)

# Import and add default/delinquency field 
ploan_api = read.csv('prosperLoanData.csv')
defdq = ifelse(ploan_api$LoanStatus %in% c("Completed", "Current", "FinalPaymenteInProgress"), 0, 1)  
ploan_api$defdq = defdq
ploan_api$defdq_fac = as.factor(ploan_api$defdq)

# Get rid of any columns with more than 30% null: 

ploan_api <- ploan_api[ ,colSums(is.na(ploan_api) | ploan_api=='') < nrow(ploan_api)*0.70]

# Run thorugh data dictionary and choose first draft of categorical and continuous variables
pcat = c('Term','ListingCategory..numeric.','BorrowerState', 'Occupation', 
         'EmploymentStatus', 'IsBorrowerHomeowner', 'IncomeVerifiable')
num_cat = length(pcat)

pcont = c('BorrowerAPR', 'ProsperRating..numeric.', 'EmploymentStatusDuration', 'CreditScoreRangeLower',
          'CurrentCreditLines','OpenCreditLines', 'OpenRevolvingAccounts', 'InquiriesLast6Months', 'CurrentDelinquencies',
          'DelinquenciesLast7Years', 'PublicRecordsLast10Years', 'DebtToIncomeRatio', 'IncomeRange',
          'StatedMonthlyIncome','LoanCurrentDaysDelinquent', 'PercentFunded', 'InvestmentFromFriendsCount', 'Investors')
num_cont=length(pcont)


#---------Refine CONTINUOUS variables------------ 

#Bucket Income Range into numbers
ir_range = as.data.frame(ploan_api$IncomeRange)
ir_range$defdq=ploan_api$defdq
colnames(ir_range) = c('IncomeRange', 'defdq')
ir_range_map = unique(ploan_api$IncomeRange)
ir_range_map = as.data.frame(ir_range_map)
ir_range_map$map = c(2, 3, '', 5, 4, 1, 0, 0)
colnames(ir_range_map)=c('IncomeRange', 'map')
ir_range$map = ir_range$IncomeRange   #creae dummy column to be replaced by mapped numbers
ir_range$map = as.character(ir_range$map)
str(ir_range)

# Create loop to map IncomeRange to number
for(i in 1:nrow(ir_range_map)) {
  irval = ir_range_map$IncomeRange[i]
  irval_map = ir_range_map$map[i]
  #print(irval)
  #print(irval_map)
  ir_range$map[which(ir_range$IncomeRange==irval)]=irval_map
}
table(ir_range$map)
ploan_api$IncomeRangeMap = ir_range$map   #throw back into to ploan and update pcont (Continous variables list)
ploan_api$IncomeRangeMap = as.numeric(ploan_api$IncomeRangeMap)
# add IncomeRangeMap to pcont & remove IncomeRange
pcont = c(pcont, 'IncomeRangeMap')
pcont = pcont[-which(pcont=='IncomeRange')]

# Create buckets for Continuos variables 

ploan_api$APR_Bucket = cut(ploan_api$BorrowerAPR, c(0, 0.15, 0.21, 0.23, 0.28, 0.52))
ploan_api$DTIBucket = cut(ploan_api$DebtToIncomeRatio, c(0, 0.14, 0.22, 0.32, 10)) 

  # Create mapping tables for APR & DTI buckets
rm(bucket.map)
bucket.map = data.frame(unique(ploan_api$APR_Bucket));  bucket.map
bucket.map$val = c(0.21, 0.15, 0.28, 0.52, 0.23, '')
colnames(bucket.map) = c('APR_Bucket','Value')
rm(ploan.stage)
ploan.stage = as.data.frame(ploan_api[,c('APR_Bucket','defdq')])
ploan.stage = merge(ploan.stage, bucket.map, by='APR_Bucket',all.x=T)
ploan_api$APR_Bucket = as.numeric(ploan.stage$Value)

ggplot(ploan_api, aes(x=APR_Bucket, y=LoanOriginalAmount, fill=defdq_fac))+
  geom_bar(stat='identity') + 
  scale_fill_manual(name='Default', breaks=c(0,1),
                    values = c('darkslategray4','indianred1'))

rm(bucket.map)
bucket.map = data.frame(unique(ploan_api$DTIBucket));  bucket.map
bucket.map$val = c(2, 1, 4, 3,'')
colnames(bucket.map) = c('DTIBucket','Value')
rm(ploan.stage)
ploan.stage = as.data.frame(ploan_api[,c('DTIBucket','defdq')])
ploan.stage = merge(ploan.stage, bucket.map, by='DTIBucket',all.x=T)
ploan_api$DTIBucket = as.numeric(ploan.stage$Value)

ggplot(ploan_api, aes(x=DTIBucket, y=LoanOriginalAmount, fill=defdq_fac))+
  geom_bar(stat='identity') + 
  scale_fill_manual(name='Default', breaks=c(0,1),
                    values = c('darkslategray4','indianred1'))


#---------Refine CATEGORICAL variables------------ 

# Categorical:  Too many categories?  
str(ploan_api[,pcat])    # too many categories for Borrower State & Occupation 

# Reduce State to Region.  use mapping file
state_map = read.csv('States_abbrev_region.csv')
colnames(state_map) = c('State', 'BorrowerState', 'Region')
ploan_api = merge(ploan_api, state_map, by='BorrowerState', all.x=T)

# Reduce Occupation by grouping all low-count jobs into 'Other'
table1 = table(ploan_api$Occupation)
table1 = as.data.frame(table1)
table1[order(table1$Freq),]
colnames(table1) = c('Occupation', 'Freq')
include.yn = ifelse(table1$Freq>(nrow(ploan_api)*0.01), 1, 0)
table1$include = include.yn
# merge and create new field 'Occ1' for new Occupation Field 
ploan_api = merge(ploan_api, table1, by='Occupation', all.x=T)
ploan_api$Freq = NULL  #remove 'Freq' column 
ploan_api$Occupation = as.character(ploan_api$Occupation)
ploan_occ1 = rep('', nrow(ploan_api))
for (i in 1:nrow(ploan_api)) {
  if (ploan_api$include[i]==1) {
    ploan_occ1[i] = ploan_api$Occupation[i]
  } else {
    ploan_occ1[i] = 'Other'
  }
}
ploan_api$Occ1 = ploan_occ1
table(ploan_api$Occ1)
# re-factor Occ1
ploan_api$Occ1 = as.factor(ploan_api$Occ1)

# add Occ1 & remove Occupation from pcat 
pcat = c(pcat, 'Occ1')  
pcat = pcat[-which(pcat=='Occupation')] 

#too many values for BorrowerState. replace with Region
pcat = pcat[-which(pcat=='BorrowerState')] 
pcat = c(pcat, 'Region')

#-- LOAN PURPOSE: Prosper categorizes loan by number in 'ListingCategory..numeric.'  Remap to og names

categories <- c("Debt Consolidation"=1, "Home Improvement"=2, "Business"=3, 
                "Student Use"=5, "Auto"=6, "Baby/Adoption"=8, 
                "Boat"=9, "Cosmetic"=10, "Engagement Ring"=11, 
                "Green Loans"=12, "Household Expenses"=13, 
                "Large Purchases"=14, "Medical/Dental"=15, "Motorcycle"=16, 
                "RV"=17, "Taxes"=18, "Vacation"=19, "Wedding"=20)

categories = as.data.frame(categories)
purpose_map = tibble::rownames_to_column(categories)  # turn rowname into column 
colnames(purpose_map) = c('Purpose', 'ListingCategory..numeric.')
purpose_map$ListingCategory..numeric. = as.integer(purpose_map$ListingCategory..numeric.); str(purpose_map)
rm(ptest)
ptest = merge(ploan_api, purpose_map, by='ListingCategory..numeric.', all.x=TRUE)
nrow(ptest); dim(ptest); head(ptest[-which(ptest$ListingCategory..numeric.==0),]) 
ploan_api$Purpose = ptest$Purpose
pcat = c(pcat, 'Purpose')
pcat = pcat[-which(pcat=='ListingCategory..numeric.')]


#--------- Univariate Analysis ------------ 

# GRID of HISTOGRAMS 

num_cont = length(pcont) 
len_pcont1= as.integer(num_cont/2)
len_pcont2 = num_cont - len_pcont1
pcont_short1 = pcont[1:len_pcont1]
pcont_short2 = pcont[(len_pcont1+1):num_cont]

#----- Grid of histograms (Continuous) -----------

chart_frame1 = list() 
for (i in 1:len_pcont1) { 
  field_name = pcont_short1[i]
  chart_frame1[[i]] = ggplot(ploan_api, aes_string(x=field_name)) + 
    geom_histogram(position='identity', fill='dodgerblue') + 
    scale_y_continuous(labels=comma)
}  
do.call(grid.arrange, c(chart_frame1, list(ncol=3, top=textGrob("Grid of Histogram (Counts): Part I"))))
rm(chart_frame1)

chart_frame2 = list()
for (i in 1:len_pcont2) { 
  field_name = pcont_short2[i]
  chart_frame2[[i]] = ggplot(ploan_api, aes_string(x=field_name)) + 
    geom_histogram(position='identity', fill='darkslategray4') + 
    scale_y_continuous(labels=comma)
}
do.call(grid.arrange, c(chart_frame2, list(ncol=3, top=textGrob("Grid of Histograms (Counts): Part II"))))
rm(chart_frame2)

#-------Grid of Pie Charts (Categorical)--------- 

# turn Purpose & State into factors
ploan_api$Purpose = as.factor(ploan_api$Purpose)
ploan_api$Term = as.factor(ploan_api$Term)
num_cat = length(pcat) 
chart_frame_cat = list()
for (i in 1:num_cat) { 
  field_name = pcat[i]
  chart_frame_cat[[i]] = ggplot(ploan_api, aes_string(x=factor(''), fill=field_name)) + 
    geom_bar() +
    xlab('Count') + 
    ylab(field_name) + 
    scale_fill_brewer(palette='GnBu', labels=comma,guide=FALSE)  +  #remove legend (doesn't fit) 
    coord_polar(theta='y')   # traditional pie slice
}
do.call(grid.arrange, c(chart_frame_cat,list(ncol=3, top=textGrob("Categorical Pie Chart"))))
rm(chart_frame_cat)

# ------------ Bivariate Analysis-------------

#------ TABLES with aggregated default amounts and estimated yields
aggregate(ploan_api$defdq, by=list(ploan_api$IncomeRangeMap), mean, na.rm=T)

#-------TABLES of aggregated default: Continuous

# plot line of default rates by bucket. create table with defaults.  create duration buckets 
bin_max = max(ploan.sub$EmploymentStatusDuration)
bin_min = min(ploan.sub$EmploymentStatusDuration)
nbucket = 12
bin_num = rep(0, nrow(ploan.sub))
bin_size = round((bin_max - bin_min)/nbucket, 0); bin_size
bin_marker = seq(bin_min, bin_max, bin_size) 
bin_marker = c(bin_marker, bin_max)
# use cut function to list out bins for each value in Employment Duration 
ploan.sub$bin_assign = cut(ploan.sub$EmploymentStatusDuration, bin_marker)
bin_buckets = aggregate(ploan.sub$defdq, by=list(ploan.sub$bin_assign), mean)
bin_buckets$bin_marker = bin_marker[1:12]

#------ Tables with aggregated default amounts and estimated yields:  Categorical variables 

rm(dt)
for (i in 1:num_cat) {
  field_name = pcat[i]
  print(field_name)
  dt = do.call(data.frame, aggregate(ploan_api$defdq, by=list(ploan_api[,field_name]), 
                                     FUN=function(x) c(mn=mean(x),n=length(x))))
  colnames(dt) = c('value','mean_default','count')
  dt$mean = as.numeric(dt$mean)
  dt=dt[order(-dt$mean),]
  dt$mean = percent(dt$mean)
  print(dt) 
}


#Continuous:  Counts:   Grid of Bivariate Histograms
num_cont = length(pcont) 
len_pcont1= as.integer(num_cont/2)
len_pcont2 = num_cont - len_pcont1
pcont_short1 = pcont[1:len_pcont1]
pcont_short2 = pcont[(len_pcont1+1):num_cont]

chart_frame1 = list() 
for (i in 1:len_pcont1) { 
  field_name = pcont_short1[i]
  chart_frame1[[i]] = ggplot(ploan_api, aes_string(x=field_name, fill=ploan_api$defdq_fac)) + 
    geom_histogram(position='identity')  + 
    scale_fill_manual(name = 'Default',
                      breaks=c(0,1), 
                      values=c('dodgerblue','indianred1'),
                      labels=c('No-Default','Default'))
}
do.call(grid.arrange, c(chart_frame1, list(ncol=3, top=textGrob("Bivariate Histogram Counts by Default: Part I"))))
rm(chart_frame1)

# Continous:  Stacked Histogram by origination amount 
chart_frame1 = list() 
for (i in 1:len_pcont1) { 
  field_name = pcont_short1[i]
  chart_frame1[[i]] = ggplot(ploan_api, aes_string(x=field_name, y=ploan_api$LoanOriginalAmount, 
                                                   fill=ploan_api$defdq_fac)) +
    geom_bar(stat='identity') + 
    xlab(field_name) + 
    ylab('Origination Amount') + 
    scale_fill_manual(name='Default', breaks=c(0,1), 
                      values=c('darkslategray4','indianred1'),
                      labels=c('No-Default','Default')) + 
    scale_y_continuous(labels=comma)
}
do.call(grid.arrange, c(chart_frame1, list(ncol=3, top=textGrob("Continuous: Bivariate Histogram by Loan Amount against Default"))))
rm(chart_frame1)

# Reduced continuous list (for presentation)
pcont_short3 = pcont_short1[-c(1, 6, 9)]; pcont_short3
pcont_short3 = c(pcont_short3, 'APR_Bucket','DTIBucket', 'IncomeRangeMap')

# Continous:  Stacked bars by origination amount 
chart_frame1 = list() 
for (i in 1:length(pcont_short3)) { 
  field_name = pcont_short3[i]
  chart_frame1[[i]] = ggplot(ploan_api, aes_string(x=field_name, y=ploan_api$LoanOriginalAmount, 
                                                   fill=ploan_api$defdq_fac)) +
    geom_bar(stat='identity') + 
    xlab(field_name) + 
    ylab('Origination Amount') + 
    scale_fill_manual(name='Default', breaks=c(0,1), 
                      values=c('darkslategray4','indianred1'),
                      labels=c('No-Default','Default')) + 
    scale_y_continuous(labels=comma)
}
do.call(grid.arrange, c(chart_frame1, list(ncol=3, top=textGrob("Continuous: Bivariate by Loan Amount against Default"))))
rm(chart_frame1)


#Continuous:  Histogram Counts
chart_frame2 = list()
for (i in 1:len_pcont1) { 
  field_name = pcont_short2[i]
  chart_frame2[[i]] = ggplot(ploan_api, aes_string(x=field_name, fill=ploan_api$defdq_fac)) + 
    geom_histogram(position='identity')  + 
    scale_fill_manual(name = 'Default',
                      breaks=c(0,1), 
                      values=c('darkseagreen4','tomato2'),
                      labels=c('No-Default','Default'))
}
do.call(grid.arrange, c(chart_frame2, list(ncol=3, top=textGrob("Bivariate Histogram Counts by Default: Part II"))))
rm(chart_frame2)

# Continous:  Stacked Histogram by origination amount:  part II 
chart_frame2 = list()
for (i in 1:len_pcont2) { 
  field_name = pcont_short2[i]
  chart_frame2[[i]] = ggplot(ploan_api, aes_string(x=field_name, y=ploan_api$LoanOriginalAmount, 
                                                   fill=ploan_api$defdq_fac)) +
    geom_bar(stat='identity') + 
    xlab(field_name) + 
    ylab('Origination Amount') + 
    scale_fill_manual(name='Default', breaks=c(0,1), 
                      values=c('darkseagreen4','tomato2'),
                      labels=c('No-Default','Default')) + 
    scale_y_continuous(labels=comma)
}
do.call(grid.arrange, c(chart_frame2, list(ncol=3, top=textGrob("Continuous: Bivariate Histogram by Loan Amount against Default: Part II"))))
rm(chart_frame2)

#Categorical Bivariate:  Grid of stacked charts (by origination amount) 

# Presentation: reduce to six categorical variables
pcat2 = pcat[-2]

rm(chart_frame1)
chart_frame1= list() 
num_cat2 = length(pcat2)
for (i in 1:num_cat2) { 
  field_name = pcat2[i]
  chart_frame1[[i]] = ggplot(ploan_api, aes_string(x=field_name, y=ploan_api$LoanOriginalAmount, 
                                                   fill=ploan_api$defdq_fac)) +
    geom_bar(stat='identity') + 
    xlab(field_name) + 
    ylab('Origination Amount') + 
    scale_fill_manual(name='Default',breaks=c(0,1), labels=c('Non-Default','Default'),
                      values=c('aquamarine4','firebrick3')) + 
    scale_y_continuous(labels=comma)
}

do.call(grid.arrange, c(chart_frame1, list(ncol=3, top=textGrob("Categorical: Bivariate vs Default by Orig Amount"))))

# Categorical;  Grid of stacked charts by count

rm(chart_frame2)
chart_frame2= list() 
for (i in 1:num_cat) { 
  field_name = pcat[i]
  chart_frame2[[i]] = ggplot(ploan_api, aes_string(x=field_name, fill=ploan_api$defdq_fac)) +
    geom_bar() + 
    xlab(field_name) + 
    ylab('Count') + 
    scale_fill_manual(name='Default',breaks=c(0,1), labels=c('Non-Default','Default'),
                      values=c('cadetblue2','coral3')) + 
    scale_y_continuous(labels=comma)
}  
do.call(grid.arrange, c(chart_frame2, list(ncol=3, top=textGrob("Categorical: Bivariate Stack Count by Default"))))


#------------CORRELATIONS-------------

#------- MULTICOLLINEARITY:  look at correlations btn variables ------ 
#--------CONTINUOUS vs. continuous:  Correlation matrix of variables

# Compute Correlation matrix
ploan_pcont = ploan_api[,pcont]
ploan_pcont = na.omit(ploan_pcont)
pcorr = round(cor(ploan_pcont), 1); pcorr
ggcorrplot(pcorr, type='lower', hc.order=TRUE)   
ggcorrplot(pcorr, type='lower', hc.order=FALSE,  #hc.order: orders graph by correlation 
           outline.col = 'deepskyblue4',   
           ggtheme=ggplot2::theme_classic,
           colors=c('coral3','white','cyan4'))   #grey out blank order. 

#------ CONTINUOUS vs. Categorical:  Point Biserial Corr with Default 

# Loop through continuous variables vs. default 
pbiserial = matrix('', num_cont, 2)
for (i in 1:num_cont) {
  field_name = pcont[i]
  pbiserial[i,1] = field_name
  pbiserial[i,2] = biserial.cor(ploan_api[,field_name], ploan_api$defdq_fac, use='complete.obs', level=1)
}
pbiserial = as.data.frame(pbiserial)
colnames(pbiserial) = c('Field','Point_Biserial_Correlation')
pbiserial$Point_Biserial_Correlation = as.numeric(pbiserial$Point_Biserial_Correlation)
# Bar plot of correlations 
ggplot(pbiserial, aes(x=Field, y=Point_Biserial_Correlation, fill=Field)) + 
  geom_bar(stat='identity') + coord_flip() + 
  # scale_fill_brewer(palette='Spectral', labels=comma)   # most palettes can't handle more than 10 colors. 
  ggtitle('Correlations against Default:  Continuous Variables')

#---------Correlations: Categorical vs. Categorical:  Cramer's V 

cramer.frame = matrix('', num_cat, 2)
n_uniqdef = length(unique(ploan_api$defdq)); n_uniqdef
for (i in 1:num_cat) {
  field_name = pcat[i]
  cramer.frame[i,1]=field_name
  n_uniq1 = length(unique(ploan_api[,field_name]))
  chi.sq.val = chisq.test(ploan_api[,field_name], ploan_api$defdq, correct=TRUE)$statistic
  cramer.frame[i,2] = round(sqrt((chi.sq.val / nrow(ploan_api)) / min(n_uniq1, n_uniqdef)), 3)
}
colnames(cramer.frame) = c('Field','CramerV')
cramer.frame = as.data.frame(cramer.frame)

ggplot(cramer.frame,aes(x=Field, y=CramerV, fill=Field)) +
  geom_bar(stat='identity') + 
  xlab('Field') + 
  ylab('CramersV Correlation') + 
  ggtitle('Correlations against Default: Categorical Variables') +
  coord_flip() + 
  scale_fill_brewer(palette='RdBu')



#--------Select Multivariate Visuals-------------

#----------- Estimatd Yields vs. Borrower Rate and Prosper Rating Scatter Plot 

rm(ploan.sub)
ploan.sub = ploan_api[, c("EstimatedEffectiveYield",'BorrowerAPR',"ProsperRating..Alpha.")]
# get rid of NAs on the rating
ploan.sub = ploan.sub[-which(ploan.sub$ProsperRating..Alpha.==''),]

ggplot(ploan.sub, aes(y = EstimatedEffectiveYield, x = BorrowerAPR, 
                      color = ProsperRating..Alpha.)) + 
  scale_colour_brewer(palette = "Spectral") +
  geom_point() + 
  labs(title = "Estimated Effective Yield vs. Borrower APR")


#------- Run through Random Forest:  Variable Importance  

# Reduce variable list
pcont2 = c('StatedMonthlyIncome','ProsperRating..numeric.', 'OpenRevolvingAccounts','IncomeRangeMap',
           'OpenCreditLines','EmploymentStatusDuration','DebtToIncomeRatio')
num_cont = length(pcont2) 
pcont_lagrid = c('ProsperRating..numeric.', 'CreditScoreRangeLower', 'OpenRevolvingAccounts',
                 'IncomeRangeMap','OpenCreditLines','EmploymentStatusDuration','PublicRecordsLast10Years')
num_contla = length(pcont_lagrid)

# Presentation:  New Bivariate Grid of defaults 

#Continuous:  Counts:   Grid of Bivariate Histograms
chart_frame1 = list() 
for (i in 1:num_cont) { 
  field_name = pcont2[i]
  chart_frame1[[i]] = ggplot(ploan_api, aes_string(x=field_name, fill=ploan_api$defdq_fac)) + 
    geom_histogram(position='identity')  + 
    scale_fill_manual(name = 'Default',
                      breaks=c(0,1), 
                      values=c('dodgerblue3','firebrick3'),
                      labels=c('No-Default','Default')) + 
    scale_y_continuous(labels=comma)
}
do.call(grid.arrange, c(chart_frame1, list(ncol=3, top=textGrob("Bivariate Histogram Counts by Default"))))


#Continuous:  Counts:   Grid of Bivariate Stacked Bars by Loan Amount 
chart_frame1 = list() 
for (i in 1:num_contla) { 
  field_name = pcont_lagrid[i]
  chart_frame1[[i]] = ggplot(ploan_api, aes_string(x=field_name, y=ploan_api$LoanOriginalAmount, 
                                                   fill=ploan_api$defdq_fac)) +
    geom_bar(stat='identity') + 
    xlab(field_name) + 
    ylab('Origination Amount') + 
    scale_fill_manual(name='Default', breaks=c(0,1), 
                      values=c('darkslategray4','deeppink4'),
                      labels=c('No-Default','Default')) + 
    scale_y_continuous(labels=comma)
}
do.call(grid.arrange, c(chart_frame1, list(ncol=3, top=textGrob("Continuous: Bivariate Histogram by Loan Amount against Default"))))

# -------- MODEL FIT-----------------
#turn cont variables to categorical 

rm(pcont2) 
pcont2 = c('ProsperRating..numeric.', 'OpenRevolvingAccounts','IncomeRangeMap',
           'EmploymentStatusDuration','DebtToIncomeRatio', 'InquiriesLast6Months',pcat)
pcont2

# Grab number of unique values per column
rm(ploan.sub) 
ploan.sub = ploan_api[, pcont2]
apply(ploan.sub, 2, function(x) length(unique(x)))   #see that Revolving Accounts, Credit Lines, Employment duration & DTI need bucketing
ploan.sub$RevBucket = cut(ploan.sub$OpenRevolvingAccounts, c(0, 4, 6, 9, 51))
ploan.sub$EmpDurBucket = cut(ploan.sub$EmploymentStatusDuration, c(0, 26, 65, 135, 755))
ploan.sub$DTIBucket = cut(ploan.sub$DebtToIncomeRatio, c(0, 0.14, 0.22, 0.32, 10)) 
ploan.sub$Inq6Bucket = cut(ploan.sub$InquiriesLast6Months, c(0, 0.5, 1.0, 2, 105)) 

# remove non-bucketed columns 
ploan.sub[,c('OpenRevolvingAccounts','EmploymentStatusDuration','DebtToIncomeRatio', 'InquiriesLast6Months' )] = NULL

# handle null values 
add999 <- function(x){
  if(is.factor(x)) return(factor(x, levels=c(levels(x), "999")))
  return(x)
}

ploan.sub <- as.data.frame(lapply(ploan.sub, add999))
test <- as.data.frame(lapply(test, add999))
ploan.sub[is.na(ploan.sub)] <- "999"

# add defdq cols 
ploan.sub$defdq = ploan_api$defdq_fac

levels(ploan.sub$Purpose) = c(levels(ploan.sub$Purpose),'999')
ploan.sub$Purpose[is.na(ploan.sub$Purpose)] = '999'

# run thru RandomForest
# Grab variable importance for entire set

rf_fit = randomForest(defdq~., data=ploan.sub, importance=TRUE, ntree=100)
varImpPlot(rf_fit, type=2)

# sample train/test split : later use k-fold CV 
indexes = sample(1:nrow(ploan.sub), size=0.2*nrow(ploan.sub))
test = ploan.sub[indexes,]
train = ploan.sub[-indexes,]
# run through RandomForest and produce ROCR curve for first set of variables. 
rf_fit2 = randomForest(defdq~., data=train, importance=TRUE, ntree=100)
rf_pred = predict(rf_fit2, test, type='prob')
comp.rf = prediction(rf_pred[,2], test$defdq)
pred.rf = performance(comp.rf, 'tpr','fpr')
pred.rf2 = performance(comp.rf, measure='auc')
auc.rf = round(pred.rf2@y.values[[1]], 3)  # pulls the AUC value
plot(pred.rf, xlim=c(0,1), ylim=c(0,1), lwd=3, col='cadetblue4')

# run Logistic Regression as a model comparison
rm(glm.fit)
glm.fit = glm(defdq~., data=train, family=binomial('logit'))
glm.pred = predict(glm.fit, test, type='response')
comp.glm = prediction(glm.pred, test$defdq)
perf.glm = performance(comp.glm, 'tpr','fpr')
perf.glm2 = performance(comp.glm, measure='auc')
auc.glm = round(perf.glm2@y.values[[1]], 3) 
plot(perf.glm, lty=2, add=TRUE, lwd=3, col='brown3')
title('ROCR Curves: RF vs. Logistic Regression')
legend('bottomright', paste(c('Random Forest, AUC=','Logistic Regression, AUC='),c(auc.rf, auc.glm)),
       col=c('cadetblue4','brown3'),
       lty=c(1,2), lwd=c(3,3))


# Re-sample train/test data set and run again 
rm(test, indexes, train)
indexes = sample(1:nrow(ploan.sub), size=0.4*nrow(ploan.sub))
test = ploan.sub[indexes,]
train = ploan.sub[-indexes,]


#---  Take out some of the variables and re-run 
ploan.sub[,c('IncomeVerifiable', 'Term', 'IsBorrowerHomeowner', 'DTIBucket', 'EmploymentStatus')] = NULL

#re-fit RF model on smaller feature set 
rf.fit3 = randomForest(defdq~., data=train, importance=TRUE, ntree=100)
rf.pred3 = predict(rf.fit3, test, type='prob')
rf.comp3 = prediction(rf.pred3[,2], test$defdq)
rf.perf3 = performance(rf.comp3, 'tpr','fpr')
rf.perfauc = performance(rf.comp3, measure='auc')
auc.rf3 = round(rf.perfauc@y.values[[1]],3)
plot(rf.perf, xlim=c(0,1), ylim=c(0,1), lwd=3, col='cadetblue4')

glm.fit3 = glm(defdq~., data=train, family=binomial('logit'))
glm.pred3 = predict(glm.fit3, test, type='response')
glm.comp3 = prediction(glm.pred3, test$defdq)
glm.perf3 = performance(glm.comp3, 'tpr','fpr')
glm.perfauc3 = performance(glm.comp3, measure='auc')
auc.glm3 = round(glm.perfauc3@y.values[[1]], 3)
auc.glm3
plot(glm.perf3, lwd=3, lty=2, col='brown3', add=TRUE)
title('ROC Curve: RF vs. Logistic: Small Set')
legend('bottomright', paste(c('Random Forest AUC =', 'Logistic Regression AUC='), 
                            c(auc.rf3, auc.glm3)), lty=c(1,2), lwd=c(3,3), col=c('cadetblue4','brown3'))

