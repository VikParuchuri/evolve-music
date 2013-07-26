setwd("~/vikparuchuri/evolve-music")

is_installed <- function(mypkg) is.element(mypkg, installed.packages()[,1])

load_or_install<-function(package_names)
{
  for(package_name in package_names)
  {
    if(!is_installed(package_name))
    {
      install.packages(package_name,repos="http://lib.stat.cmu.edu/R/CRAN")
    }
    options(java.parameters = "-Xmx8g")
    library(package_name,character.only=TRUE,quietly=TRUE,verbose=FALSE)
  }
}

load_or_install(c("RJSONIO","ggplot2","stringr","foreach","wordcloud","lsa","MASS","openNLP","tm","fastmatch","reshape","openNLPmodels.en",'e1071','gridExtra'))

features = read.csv("stored_data/visualize.csv",stringsAsFactors=FALSE)

non_predictors = c("label","fs","enc","fname","label_code")

names(features)[(ncol(features)-4):ncol(features)] = non_predictors

features$label_code = as.numeric(as.factor(features$label))
feature_names = names(features)[!names(features) %in% c(non_predictors,"X")]

for(f in feature_names){
  features[,f] = as.numeric(features[,f])
}

scaled_data = scale(features[,feature_names])
scaled_data = apply(scaled_data,2,function(x) {
  x[is.na(x)] = -1
  x
})
svd_train<-svd(scaled_data,2)$u

newtrain<-data.frame(x=svd_train[,1],y=svd_train[,2],label_code=features$label_code,label=features$label)

#model = svm(score ~ x + y, data = newtrain)
#plot(model,newtrain)

collapse_frame = do.call(rbind,by(features[,feature_names],feature$label_code,function(x) apply(x,2,mean)))
line_count = tapply(tf$result_label,tf$result_label,length)
scaled_data = scale(collapse_frame)
scaled_data = apply(scaled_data,2,function(x) {
  x[is.na(x)] = -1
  x
})


svd_train<-data.frame(svd(scaled_data,2)$u,line_count=line_count,label=rownames(line_count))
svd_train <- svd_train[svd_train$X1<mean(svd_train$X1)+1.4*sd(svd_train$X1) & svd_train$X1>mean(svd_train$X1)-1.4*sd(svd_train$X1),]
svd_train <- svd_train[svd_train$X2<mean(svd_train$X2)+1.4*sd(svd_train$X2) & svd_train$X2>mean(svd_train$X2)-1.4*sd(svd_train$X2),]

p <- ggplot(newtrain, aes(x, y))
p = p + geom_point(aes(colour =label_code)) + scale_size_area(max_size=20)
p = p +   theme(axis.line = element_blank(),
                panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(),
                panel.border = element_blank(),
                axis.title.x = element_blank(),
                axis.title.y = element_blank(),
                axis.ticks=element_blank(),
                axis.text.x = element_blank(),
                axis.text.y = element_blank()) 
p = p +labs(colour="Type of Music")
p

