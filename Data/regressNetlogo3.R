# a function to run hedonic analysis on a subset of transaction data
runRegression1<-function(File1,np){
    ## sink(paste(workingDir,"r.out",sep=""),append=T); ## record all output
    ## sink(paste(workingDir,"r.out",sep="")); ## record all output
    ## print(workingDir);
    ## print(File1);
    ## print("OBJECTS:");
    ## print(ls());
    df0<-read.csv(File1)                                      #reads the file with transaction data
    ## print("ORIGINAL TABLE")
    str(df0, max.level = -1)                                  #use max.level to display only the # of records and vars  

    #1st selction: select only the rows that record actual transactions (tradecodes 1,2,3,5), thus excluding fallied trade    
    df1 <- subset(df0,tradeCode == 1 | tradeCode == 2 | tradeCode == 3 | tradeCode == 5)
    print("TABLE WITH ONLY POSITIVE TRADE CODES")
    str(df1, max.level = -1)
    
    time.max<-max(unique(df1[,"ticks"]))                      #find what was the last trade period
    write.table(data.frame(tick=time.max,nrow=nrow(df0)),
                            file="c:/usr/proj/ut/did/nrow.dat",
                            append=T,quote=F,sep=",",col.names=F,row.names=F);
    #consider adding a check if time.max = 12 (1 year) => set time.max =12 so that later on regression will not be done for transactions later that 1 year old
    
#   #debugging
#   tc1.un<-unique(df0[,"tradeCode"])                       #extracts unique value of tradeCodes, for debugging
#   tc1.un<-sort(tc1.un, decreasing = FALSE)
#   print("Trade codes, original:")
#   print(tc1.un)  
#   tc2.un<-unique(df1[,"tradeCode"])                       #extracts unique value of tradeCodes, for debugging
#   tc2.un<-sort(tc2.un, decreasing = FALSE)
#   print("Trade codes, selection:")
#   print(tc2.un)
    
    #Add missing fields
    df1$FP100<- 0     
    df1$FP100[df1$probFlood == 0.01] <- 1
    df1$FP500<- 0     
    df1$FP500[df1$probFlood == 0.002] <- 1 
    #str(df1)
    
    #Run regression for the full dataset
    #                    2         3              4       5          6        7            8              9
    lm0<-lm(log(trPr)~(bathrm + I(bathrm ^ 2) + age + I(age ^ 2) + sqft + I(sqft ^ 2) + lotsize + I(lotsize ^ 2)
    #                       10           11      12      13           14              15            16  
                                            + newHome + postFirm + FP100 + FP500 + coastalFront + log(distAmen) + log(distCBD) + 
    #                       17                   18        19         20
                                             + log(distHwy) + log(distPark) + town1M + town15B), df1)
    lm0.sum<-summary(lm0)
    print("REGRESSION, ALL TRANSACTIONS")
    print(lm0.sum)
    co <- coef(lm0)                                          #extract regression coefficients
    coNames <- names(co)
#   print("REGRESSION COEFFICIENTS, ALL TRANSACTIONS WITH NA")
#   print(co)
    co <- ifelse(is.na(co),0,co)                             #if there are coefficents with NA values - make them equal to zero
    co.std <- summary(lm0)$coefficients[,2]
    co.pv <- summary(lm0)$coefficients[,4]
    stp.list0 <- list(co.std,co.pv)
    MtrPr0 <- mean(df1$trPr,na.rm=T)
    r.lm0.sum <- lm0.sum$r.squared                           #get R value  
    regCode <- 0
    listRes0 <- c(time.max,                                  #record the time step
                                round(co[1],9),                            #intercept   
                                round(co[2],9),                            #bathrm 
                                round(co[3],9),                            #bathrm^2
                                round(co[4],9),                            #age 
                                round(co[5],9),                            #age^2 
                                round(co[6],9),                            #sqft 
                                round(co[7],9),                            #sqft^2 
                                round(co[8],9),                            #lotsize 
                                round(co[9],9),                            #lotsize^2
                                round(co[10],9),                           #newhome
                                round(co[11],9),                           #postFirm
                                round(co[12],9),                           #FP100 
                                round(co[13],9),                           #FP500
                                round(co[14],9),                           #coastalFront
                                round(co[15],9),                           #log(distAmen) 
                                round(co[16],9),                           #log(distCBD)
                                round(co[17],9),                           #log(distHwy)  
                                round(co[18],9),                           #log(distPark) 
                                round(co[19],9),                           #town1M 
                                round(co[20],9),                           #town15B 
                                round(lm0.sum$r.squared,9),
                                regCode,
                                0
                             )
    print("REGRESSION COEFFICIENTS, ALL TRANSACTIONS")
    print(listRes0)

    if (time.max >=np) {                                 # if there are transactions for more than N months, take the last N months into regresison analysis
        loop <- TRUE
        i <- (np - 1)                                      # for i < N-1
        while(loop) {
            df2 <- subset(df1,ticks >= (time.max - i))       #select only transactions for last month + i number of months earlier
             #debugging
            time.un<-unique(df2[,"ticks"])                   #extracts unique value of trade periods, i.e. ticks
            print("Number of trade months")
            print(time.un)
            #Run regression for the filtered dataset
            #                    2         3              4       5          6        7            8              9
            lm1<-try(lm(log(trPr)~(bathrm + I(bathrm ^ 2) + age + I(age ^ 2) + sqft + I(sqft ^ 2) + lotsize + I(lotsize ^ 2)
                                                 #                       10           11      12      13           14              15            16  
                                                 + newHome + postFirm + FP100 + FP500 + coastalFront + log(distAmen) + log(distCBD) + 
                                                     #                       17                   18        19         20
                                                     + log(distHwy) + log(distPark) + town1M + town15B), df2),silent=T) 
            if (!inherits(lm1, "try-error")) {                        #check if there are enough transactions to estiamte this regresison
                regCode <- 1
                lm1.sum<-summary(lm1)
                print("REGRESSION, LAST FEW MONTHS TRANSACTIONS")
                print(lm1.sum)
                co <- coef(lm1)                                          #extract regression coefficients
                co <- ifelse(is.na(co),0,co)                             #if there are coefficents with NA values - make them equal to zero
                co.std <- summary(lm1)$coefficients[,2]
                co.pv <- summary(lm1)$coefficients[,4]
                stp.list <- list(co.std,co.pv)
                MtrPr <- mean(df2$trPr,na.rm=T)
                r.lm1.sum <- lm1.sum$r.squared                           #get R value  
                listRes <- c(time.max,                                   #record the time step
                                            round(co[1],9),                            #intercept   
                                            round(co[2],9),                            #bathrm 
                                            round(co[3],9),                            #bathrm^2
                                            round(co[4],9),                            #age 
                                            round(co[5],9),                            #age^2 
                                            round(co[6],9),                            #sqft 
                                            round(co[7],9),                            #sqft^2 
                                            round(co[8],9),                            #lotsize 
                                            round(co[9],9),                            #lotsize^2
                                            round(co[10],9),                           #newhome
                                            round(co[11],9),                           #postFirm
                                            round(co[12],9),                           #FP100 
                                            round(co[13],9),                           #FP500
                                            round(co[14],9),                           #coastalFront
                                            round(co[15],9),                           #log(distAmen) 
                                            round(co[16],9),                           #log(distCBD)
                                            round(co[17],9),                           #log(distHwy)  
                                            round(co[18],9),                           #log(distPark) 
                                            round(co[19],9),                           #town1M 
                                            round(co[20],9),                           #town15B 
                                            round(lm1.sum$r.squared,9),
                                         regCode,
                                         i
                                         )
                loop <- FALSE                                            #if there are enough transactions to estimate this regresison, then the loop stops
                
                l <- lm1
                
            } else {                                                   #if there are not enough transactions to estimate this regresison, then the one more month is added the loop is repeated
                i <- i + 1
                if (time.max < i+1 | i > 11) {                                    #if i-counter reached the max number of months, then take the results of the full dataset regression 
                    loop <- FALSE
                    listRes <- listRes0
                    stp.list <- stp.list0
                    MtrPr <- MtrPr0
                }
            }
        }
    } else {
        listRes <- listRes0                                     #if fitlered dataset continues to return error (not enough variation to estiamte the regression), then take the results of the full dataset regression 
        stp.list <- stp.list0
        MtrPr <- MtrPr0
    }
    # writing the output csv
    m <- matrix(nr=1,nc=65)
    NAMES <- c()
    m[1,1] <- time.max
    NAMES <- c(NAMES,"time")
    st_Names <- paste("std_",coNames,sep="")
    p_Names <- paste("P_",coNames,sep="")
    j <- 2
    for (i in 1:20) {
        m[1,j] <- listRes[[(i+1)]]
        j <- j+1
        m[1,j] <- stp.list[[1]][i]
        j <- j+1
        m[1,j] <- stp.list[[2]][i]
        j <- j+1
        NAMES <- c(NAMES,coNames[i])
        NAMES <- c(NAMES,st_Names[i])
        NAMES <- c(NAMES,p_Names[i])
    }
    NAMES <- c(NAMES,c("R2","regCode","i","meanTrPr"))
    m[1,62:64] <- c(listRes[[22]],listRes[[23]],listRes[[24]])
    m[1,65] <- MtrPr
    outdf <- data.frame(m)
    colnames(outdf) <- NAMES

    routfile = paste(workingDir,"R_Output.csv",sep="");
    if (time.max == 1) {
##    write.csv(outdf,"~/Documents/Work/1University/Program_code/VENI/veniR/R_Output.csv",row.names=F)
        ## write.csv(outdf,"c:/usr/proj/ut/did/R_Output.csv",row.names=F)
     write.csv(outdf,routfile,row.names=F)
    } else{
##    existingCSV <- read.csv("~/Documents/Work/1University/Program_code/VENI/veniR/R_Output.csv",sep=',')
        existingCSV <- read.csv(routfile,sep=',')
        names(existingCSV) <- NAMES
        outdf <- data.frame(rbind(existingCSV,outdf))
##    write.csv(outdf,"~/Documents/Work/1University/Program_code/VENI/veniR/R_Output.csv",row.names=F)
        ## write.csv(outdf,"c:/usr/proj/ut/did/R_Output.csv",row.names=F)
        write.csv(outdf,routfile,row.names=F)
    }
    ## jl:
    ## sink(paste(workingDir,"nFile.out",sep=''));
    ## cat("workingDir = '",workingDir,"'\n");
    ## printf(df0[1:10,]);
    ## sink(NULL);
    return(listRes)
}

# nFile <- "~/Documents/Work/1University/Program_code/VENI/veniR/trade.csv"
# nFile1 <- "~/Documents/Work/1University/Program_code/VENI/veniR/trade1.csv"
# #nFile <- "~/Documents/Work/1University/Program_code/VENI/veniNtl/trade.csv"
# #nFile <- "~/Documents/Work/1University/Program_code/VENI/veniNtl/tradeAll.csv"
# df <- read.csv(nFile)
# head(df)
# File1 <- nFile
# df <- df[which(df$ticks == 1),]
# write.csv(df,'trade1.csv',row.names=F)
# rr <- runRegression1(nFile1)
rResult <- runRegression1(nFile,np)
## rResult <- list(1,2,3,4); ## test
## rResult
print(rResult)
