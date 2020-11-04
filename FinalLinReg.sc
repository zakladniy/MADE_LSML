// -----------------------Libraries import------------------------ //

import breeze.linalg._
import breeze.stats._
import breeze.numerics._
import java.io._
import scala.util.control.Breaks.{break, breakable}

// ----------------------------Functions------------------------ //

// Read data from csv-file
def dataLoad(path: String): DenseMatrix[Double] = {
  return csvread(file = new File(path), ',', skipLines=1)
}

// Write predictions to csv-file
def dataDump(path: String, data: DenseVector[Double]): Unit = {
  val dataMatrix: DenseMatrix[Double] = DenseMatrix(data).t
  return csvwrite(file = new File(path), dataMatrix, ',')
}

// Write train and validations results to txt-file
def writeMetricToFile(metricName: String, trainMetric: Double, validMetric: Double) = {
  val pw = new PrintWriter(new File("C:\\Users\\User\\Desktop\\HW5_Scala_new\\train_val_metric.txt"))
  pw.write(("Train and validation " + metricName + " metric" + "\n\n"))
  pw.write(("Train " + metricName + ": " + trainMetric.toString + "\n"))
  pw.write(("Validation " + metricName + ": " + validMetric.toString + "\n"))
  pw.close()
}

// Add ones vector to data
def getXWithBias(original: DenseMatrix[Double]): DenseMatrix[Double] = {
  val ones: DenseMatrix[Double] = DenseMatrix.ones(original.rows, 1)
  val dataWithOnes = ones.data ++ original.data
  return DenseMatrix.create(original.rows, original.cols + 1, dataWithOnes)
}

// Matrix statistics calculation
def maxtrixStatistics(x: DenseMatrix[Double]): (DenseVector[Double], DenseVector[Double]) = {
  return (mean(x(::, *)).t, stddev(x(::, *)).t)
}

// Split data into train and validation datasets
def dataSplit(x: DenseMatrix[Double], testSize: Double = 0.5): (DenseMatrix[Double],
  DenseMatrix[Double], DenseVector[Double], DenseVector[Double]) = {
  val y: DenseVector[Double] = x(::, 0)
  val t = x(::, 1 to -1)
  val H = DenseMatrix.zeros[Double](t.rows, t.cols)
  val matrMean = H(*, ::) + maxtrixStatistics(t)._1
  val matrStd = H(*, ::) + maxtrixStatistics(t)._2
  val dataStand = (t - matrMean) / matrStd
  val testLen: Int = (testSize * t.rows).toInt
  val trainLen: Int = t.rows - testLen
  val xTrain: DenseMatrix[Double] = dataStand(0 until trainLen, ::)
  val yTrain: DenseVector[Double] = y(0 until trainLen)
  val xValid: DenseMatrix[Double] = dataStand(trainLen to -1, ::)
  val yValid: DenseVector[Double] = y(trainLen to -1)
  csvwrite(file = new File("C:\\Users\\User\\Desktop\\HW5_Scala_new\\xValid.csv"),  xValid, ',')
  csvwrite(file = new File("C:\\Users\\User\\Desktop\\HW5_Scala_new\\xTrain.csv"),  xTrain, ',')
  return (xTrain, xValid, yTrain, yValid)
}

// Data preparation for predict unknown data
def dataPrep(x: DenseMatrix[Double], mean: DenseVector[Double], std: DenseVector[Double]): DenseMatrix[Double] = {
  val H = DenseMatrix.zeros[Double](x.rows, x.cols)
  val matrMean = H(*, ::) + mean
  val matrStd = H(*, ::) + std
  return (x - matrMean) / matrStd
}

// --------------------------------------Classes--------------------------- //
class RegressionMetrics() {

  // R2-score metric
  def r2Score(yTrue: DenseVector[Double], yPred: DenseVector[Double]): Double = {
    val sst: Double = sum((yTrue - mean(yTrue)) * (yTrue - mean(yTrue)))
    val ssr: Double = sum((yPred - yTrue) * (yPred - yTrue))
    return 1 - (ssr / sst)
  }

  // RMSE
  def rmseScore(yTrue: DenseVector[Double], yPred: DenseVector[Double]): Double = {
    return sqrt(mean(yPred - yTrue) * mean(yPred - yTrue))
  }

  // MSE
  def mseScore(yTrue: DenseVector[Double], yPred: DenseVector[Double]): Double = {
    return (mean(yPred - yTrue) * mean(yPred - yTrue))
  }
}

class LinearRegression() {

  // Added X-matrix by ones
  def getXWithBias(original: DenseMatrix[Double]): DenseMatrix[Double] = {
    val ones: DenseMatrix[Double] = DenseMatrix.ones(original.rows, 1)
    val dataWithOnes = ones.data ++ original.data
    return DenseMatrix.create(original.rows, original.cols + 1, dataWithOnes)
  }

  // Loss function for coefficients optimization
  def lossFunction(x: DenseMatrix[Double], y: DenseVector[Double],
                   B: DenseVector[Double]): Double = {
    return sum((x * B - y) * (x * B - y)) / (2 * x.rows)
  }

  // Train linear regression
  def fit(x: DenseMatrix[Double], y: DenseVector[Double],
          iterations: Int=10000, lr: Double=0.01, eps: Double=0.05): (DenseVector[Double], DenseVector[Double]) = {
    var lossHistory : DenseVector[Double] = DenseVector.zeros(iterations)
    var loss: Double = 0
    var B: DenseVector[Double] = DenseVector.ones(x.cols + 1)
    val pw = new PrintWriter(new File("C:\\Users\\User\\Desktop\\HW5_Scala_new\\fit_report.txt"))
    pw.write(("Fitting loss:" + "\n"))
    breakable { for (iteration <- 0 to iterations - 1){
      B = B - (lr / getXWithBias(x).rows) * (getXWithBias(x).t * (getXWithBias(x) * B - y))
      loss = lossFunction(getXWithBias(x), y, B)
      pw.print(iteration.toString + ": " + loss.toString + "\n")
      if (abs(loss - lossHistory(iteration)) <= eps) {
        pw.print(("Fit end at iteration " + iteration.toString + " with loss " + loss.toString))
        break
      }
      lossHistory(iteration) = loss
    }
      break
    }
    pw.close
    return (B, lossHistory)
  }

  // Predict values
  def predict(x: DenseMatrix[Double], B: DenseVector[Double]): DenseVector[Double] = {
    return getXWithBias(x) * B
  }
}


// -------------------------------Program work----------------------- //

// Data load
val trainDataMatrix  = dataLoad(("C:\\Users\\User\\Desktop\\HW5_Scala_new\\train_data.csv"))


// Data separation for train and valid data
val (xTrain, xValid, yTrain, yValid) = dataSplit(trainDataMatrix)
val xValid_ = csvread(file = new File("C:\\Users\\User\\Desktop\\HW5_Scala_new\\xValid.csv"))


// Train linear regression
val linReg = new LinearRegression()
val (bNew, lossHistory) = linReg.fit(xTrain, yTrain)


// Estimate linear regression
val metric = new RegressionMetrics()
val trainR2_ = metric.r2Score(yTrain, linReg.predict(xTrain, bNew))
val validR2_ = metric.r2Score(yValid, linReg.predict(xValid_, bNew))


// Write metrics to txt-file
writeMetricToFile("R2", trainR2_, validR2_)


// Load test data
val testDataMatrix  = dataLoad(("C:\\Users\\User\\Desktop\\HW5_Scala_new\\test_data.csv"))


// Preparation test data for predict
val (trainMean, trainStd) = maxtrixStatistics(trainDataMatrix(::, 1 to -1))
val dataTest = dataPrep(testDataMatrix, trainMean, trainStd)


// Predict test values
val yTestPred = linReg.predict(dataTest, bNew)


// Write prediction to csv-file on disk
dataDump("C:\\Users\\User\\Desktop\\HW5_Scala_new\\test_preds.csv", yTestPred)