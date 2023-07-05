##Rotational Momentum Trading Program

*************Disclaimer*****************
This program is a novelty project and is not to be used for trading in the stock market. I am not liable for any financial losses that may occur from the use of this program. 


#Background:

General
Momentum is a measured phenomena in the stock market. Stocks that are increasing in value, have the tendancy to continue to increase in value; and stocks that are decreasing in value tend to continue to decrease in value. So it is possible to take advantage of this "momentum". It has been measured that the momentum will change every 15 days, so at this point it will be good time to check which stock has the highest momentum, and re-invest your assets; and choose the ETF doing the best of the previous quarter.  

In this porgram we will use sector specific ETFs, because there is usually always at least one sector doing well, even if other sectors are doing poorly; because the sectors are not related. So this way we can choose the sector that is doing the best every re-investment period. 

US sector specific ETFs = ["FDN", "IBB", "IEZ", "IGV", "IHE", "IHF", "IHI", "ITA", "ITB", "IYJ", "IYT", "IYW", 	"IYZ", "KBE", "KCE", "KIE", "PBJ", "PBS", "SMH", "VNQ", "SHY"]

Canadian sector specific ETFs (not as good) = canada_etfs = ["FDN.TO", "FBT.TO", "ZEO.TO", "SKYY.TO", "XHC.TO", "FHH.TO", "ZGEN.TO", "EARK.NE", "CIF.TO", "FHG.TO", "TRVL.TO", "FHQ.TO", "NXTG.TO", "ZBK.TO", "RUBY.TO", "FLI.TO", "XST.TO", "COMM.TO", "CHPS.TO", "TGRE.TO", "XGB.TO"]

#Portfolio choices
Now to make the portfolio less volatile we will add Bonds as well, and do the same thing. Take the bond with the best momentum for that period. Depending on the level of risk you want to take on more bonds (less risk). common porfolio splits are:
- 80% sector ETFs, 20% Bond ETFs
- 70% sector ETFs, 30% Bond ETFs
- 60% sector ETFs, 40% Bond ETFs

US bond ETFs = ["BIL", "TIP", "IEI", "IEF", "TLH", "TLT", "SHY"]
Canadian bond ETFs = ["VSC.TO", "VSB.TO", "ZIC.TO", "HBB.TO", "TDB.TO", "XBB.TO", "XGB.TO"]

#Sharpe Ratio Surface
So there is a caviat to the lookback and holding period. That is, that 15 days is not always the best holding period and 3 months is not alwas 



#Steps to run the program. 

1. Download data from yahoo finance
    - Run the DataIngestor object to get the data

2. Run the SharpeSurfaceGenerator and choose the best holding and lookback periods

3. Run the ETFRanker with the holding and lookback periods to find the highest ranked ETF