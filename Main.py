import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def readCSV(filepath):
    df = pd.read_csv(filepath)
    return df


# Generating the pair plot for basic information
def pairPlot(dataFrame):
    pairPlotImg = sns.pairplot(dataFrame)
    pairPlotImg.savefig("Plots/rawPairPlotImg.png")


def plotRegression(dataFrame):
    # Regression between x and y dimensions
    regXY = sns.lmplot(x="x", y="y", data=dataFrame, scatter_kws={"s": 24}, line_kws={"color": "red"})
    regXY.fig.suptitle("Regression between X and Y dimensions", fontsize=8, ha="center")
    regXY.set_xlabels("X Dimension")
    regXY.set_ylabels("Y Dimension")
    regXY.savefig("Plots/regXY.png")

    # Regression between x and z dimensions
    regXZ = sns.lmplot(x="x", y="z", data=dataFrame, scatter_kws={"s": 24}, line_kws={"color": "red"})
    regXZ.fig.suptitle("Regression between X and Z dimensions", fontsize=8, ha="center")
    regXZ.set_xlabels("X Dimension")
    regXZ.set_ylabels("Z Dimension")
    regXZ.savefig("Plots/regXZ.png")

    # Regression between y and z dimensions
    regYZ = sns.lmplot(x="y", y="z", data=dataFrame, scatter_kws={"s": 24}, line_kws={"color": "red"})
    regYZ.fig.suptitle("Regression between Y and Z dimensions", fontsize=8, ha="center")
    regYZ.set_xlabels("Y Dimension")
    regYZ.set_ylabels("Z Dimension")
    regYZ.savefig("Plots/regYZ")

    # Regression between y and Carat
    regYCarat = sns.lmplot(x="y", y="carat", data=dataFrame, scatter_kws={"s": 24}, line_kws={"color": "red"})
    regYCarat.fig.suptitle("Regression between Y Dimension and Carat", fontsize=8, ha="center")
    regYCarat.set_xlabels("Y Dimension")
    regYCarat.set_ylabels("Carat")
    regYCarat.savefig("Plots/regYCarat.png")

    # Regression between z and Carat
    regZCarat = sns.lmplot(x="z", y="carat", data=dataFrame, scatter_kws={"s": 24}, line_kws={"color": "red"})
    regZCarat.fig.suptitle("Regression between Z Dimension and Carat", fontsize=8, ha="center")
    regZCarat.set_xlabels("Z Dimension")
    regZCarat.set_ylabels("Carat")
    regZCarat.savefig("Plots/regZCarat.png")

    # Regression between Depth and Y
    regYDepth = sns.lmplot(x="depth", y="y", data=dataFrame, scatter_kws={"s": 24}, line_kws={"color": "red"})
    regYDepth.fig.suptitle("Regression between Depth and Y Dimension", fontsize=8, ha="center")
    regYDepth.set_ylabels("Y Dimension")
    regYDepth.set_xlabels("Depth")
    regYDepth.savefig("Plots/regYDepth.png")

    # Regression between Depth and Z
    regZDepth = sns.lmplot(x="depth", y="z", data=dataFrame, scatter_kws={"s": 24}, line_kws={"color": "red"})
    regZDepth.fig.suptitle("Regression between Depth adn Z Dimension", fontsize=8, ha="center")
    regZDepth.set_ylabels("Z Dimension")
    regZDepth.set_xlabels("Depth")
    regZDepth.savefig("Plots/regZDepth.png")

    # Regression between y and Table
    regYTable = sns.lmplot(x="table", y="y", data=dataFrame, scatter_kws={"s": 24}, line_kws={"color": "red"})
    regYTable.fig.suptitle("Table and Y Dimension", fontsize=8, ha="center")
    regYTable.set_ylabels("Y Dimension")
    regYTable.set_xlabels("Table")
    regYTable.savefig("Plots/regYTable.png")

    # Regression between z and Table
    regZTable = sns.lmplot(x="table", y="z", data=dataFrame, scatter_kws={"s": 24}, line_kws={"color": "red"})
    regZTable.fig.suptitle("Table and Z Dimension", fontsize=8, ha="center")
    regZTable.set_ylabels("Z Dimension")
    regZTable.set_xlabels("Table")
    regZTable.savefig("Plots/regZTable.png")

    # Regression between y and Price
    regYPrice = sns.lmplot(x="y", y="price", data=dataFrame, scatter_kws={"s": 24}, line_kws={"color": "red"})
    regYPrice.fig.suptitle("Y Dimension and Price", fontsize=8, ha="center")
    regYPrice.set_xlabels("Y Dimension")
    regYPrice.set_ylabels("Price")
    regYPrice.savefig("Plots/regYPrice.png")

    # Regression between z and Price
    regZPrice = sns.lmplot(x="z", y="price", data=dataFrame, scatter_kws={"s": 24}, line_kws={"color": "red"})
    regZPrice.fig.suptitle("Z Dimension and Price", fontsize=8, ha="center")
    regZPrice.set_xlabels("Z Dimension")
    regZPrice.set_ylabels("Price")
    regZPrice.savefig("Plots/regZPrice.png")

    # Regression between price and table
    regPriceTable = sns.lmplot(x="price", y="table", data=dataFrame, scatter_kws={"s": 24}, line_kws={"color": "red"})
    regPriceTable.fig.suptitle("Price and Table", fontsize=8, ha="center")
    regPriceTable.set_xlabels("Price")
    regPriceTable.set_ylabels("Table")
    regPriceTable.savefig("Plots/regPriceTable.png")

    # Regression between price and depth
    regPriceDepth = sns.lmplot(x="price", y="depth", data=dataFrame, scatter_kws={"s": 24}, line_kws={"color": "red"})
    regPriceDepth.fig.suptitle("Price and Depth", fontsize=8, ha="center")
    regPriceDepth.set_xlabels("Price")
    regPriceDepth.set_ylabels("Depth")
    regPriceDepth.savefig("Plots/regPriceDepth.png")


def cleanDF(dataFrame):
    # Removing the Index column
    dataFrame = dataFrame.drop(["Unnamed: 0"], axis=1)

    # Removing absurd dimensions
    dataFrame = dataFrame.drop(dataFrame[dataFrame["x"] <= 0].index)
    dataFrame = dataFrame.drop(dataFrame[dataFrame["y"] <= 0].index)
    dataFrame = dataFrame.drop(dataFrame[dataFrame["z"] <= 0].index)

    # Dropping the duplicate and empty data
    dataFrame.drop_duplicates(inplace=True)
    dataFrame.dropna(inplace=True)

    # Cleaning Outliers
    pairPlot(dataFrame)
    plotRegression(dataFrame)
    dataFrame = dataFrame[dataFrame["y"] < 20]
    dataFrame = dataFrame[((dataFrame["z"] < 10) & (dataFrame["z"] > 2))]
    dataFrame = dataFrame[(dataFrame["depth"] < 75) & (dataFrame["depth"] > 50)]
    dataFrame = dataFrame[(dataFrame["table"] < 80) & (dataFrame["table"] > 40)]

    # Final Pair plot
    cleanPlot = sns.pairplot(dataFrame, hue="cut")
    cleanPlot.savefig("Plots/cleanPairPlot.png")

    return dataFrame


def setFormat(dataFrame):
    dataFrame["carat"] = pd.to_numeric(dataFrame["carat"])
    dataFrame["depth"] = pd.to_numeric(dataFrame["depth"])
    dataFrame["table"] = pd.to_numeric(dataFrame["table"])
    dataFrame["price"] = pd.to_numeric(dataFrame["price"])
    dataFrame["x"] = pd.to_numeric(dataFrame["x"])
    dataFrame["y"] = pd.to_numeric(dataFrame["y"])
    dataFrame["z"] = pd.to_numeric(dataFrame["z"])

    return dataFrame


def describe(dataFrame):
    centerText = "Data's Source Information"
    centerText = centerText.center(terminalWidth)
    print(centerText)
    centerText = "(Source: Kaggle's Diamond Dataset)"
    centerText = centerText.center(terminalWidth)
    print(centerText)
    print(line)
    print("\033[1mDataframe Columns:\033[0m \n")
    print("\033[1mPrice\033[0m range in US dollars: $326--$18,823")
    print("\033[1mCarat\033[0m weight of the diamond: 0.2--5.01")
    print("\033[1mCut\033[0m quality: Fair, Good, Very Good, Premium, Ideal")
    print("\033[1mDiamond color\033[0m: J (worst) to D (best)")
    print("\033[1mClarity\033[0m: I1 (worst) to IF (best)")
    print("\033[1mLength\033[0m in mm: 0--10.74")
    print("\033[1mWidth\033[0m in mm: 0--58.9")
    print("\033[1mDepth\033[0m in mm: 0--31.8")
    print("\033[1mTotal depth percentage\033[0m: 43--79")
    print("\033[1mTable width\033[0m: 43--95")
    print(line)
    centerText = "Sample of the Dataframe\n"
    centerText = centerText.center(terminalWidth)
    print(centerText)
    print(dataFrame.head())
    print(f"\n\033[1mDataframe Dimensions\033[0m: {dataFrame.shape[0]} x {dataFrame.shape[1]}")


def violinPlot(dataframe):
    # Price vs Cut plot
    cutPlot = sns.violinplot(x="cut", y="price", data=dataframe)
    plt.suptitle("Price vs Cut", fontsize=8, ha="center")
    cutPlot.set_xlabel("Cut")
    cutPlot.set_ylabel("Price")
    plt.savefig("Plots/cutPlot.png")
    plt.close()

    # Price vs Color plot
    colorPlot = sns.violinplot(x="color", y="price", data=dataframe)
    plt.suptitle("Color vs Price", fontsize=8, ha="center")
    colorPlot.set_xlabel("Color")
    colorPlot.set_ylabel("Price")
    plt.savefig("Plots/colorPlot.png")
    plt.close()

    # Price vs Clarity plot
    clarityPlot = sns.violinplot(x="clarity", y="price", data=dataframe)
    plt.suptitle("Clarity vs Price", fontsize=8, ha="center")
    clarityPlot.set_xlabel("Clarity")
    clarityPlot.set_ylabel("Price")
    plt.savefig("Plots/clarityPlot.png")
    plt.close()


def encode(dataframe):
    encLabel = LabelEncoder()
    for column in ['cut', 'color', 'clarity']:
        dataframe[column] = encLabel.fit_transform(dataframe[column])
    return dataframe


def corMatrix(dataframe):
    pallet = sns.diverging_palette(120, 0, s=80, l=70, n=6, as_cmap=True)
    matrix = dataframe.corr()
    plt.subplots(figsize=(11, 11))
    sns.heatmap(matrix, cmap=pallet, annot=True, )
    plt.savefig("Plots/heatmap.png")


def main():
    dataFrame = (readCSV(csvFilePath))
    describe(dataFrame)
    dataFrame = setFormat(cleanDF(dataFrame))
    violinPlot(dataFrame)
    encode(dataFrame)
    describe(dataFrame)
    corMatrix(dataFrame)


if __name__ == "__main__":
    # Global Variables
    csvFilePath = "/Users/ayush/Documents/Coding/Projects/Diamond Price Analysis/diamonds.csv"
    terminalWidth = shutil.get_terminal_size().columns
    line = "-------------------------*--------------------------\n"
    line = line.center(terminalWidth)

    main()
