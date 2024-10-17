using System.Collections.Generic;

namespace MultinomialClassification {
    //1エポック2分30秒
    class DeepLearning {
        public static class Constants {
            public const double LearningRate = 0.1;   // 学習率
            public const int Epochs = 100;           // エポック数
            public const int HiddenLayerSize = 256;         // 隠れ層のサイズ

            public const int InputSize = 785;           //入力層(バイアス込み)
            public const int OutputSize = 10;            //出力層
            public const int BatchSize = 100;
        }
        public class NumberData {
            public double[] pixel_data=new double[Constants.InputSize];

            public int Label { get; set; }

            // コンストラクタ
            public NumberData(string[] data) {
                //ダミーの追加
                pixel_data[0] = 1;
                for (int i = 1; i < data.Length; i++) {
                    pixel_data[i] = double.Parse(data[i]);
                }
                Label= int.Parse(data[0]);
            }

            // 特徴量の配列を返すメソッド
            public double[] GetFeatures() {
                return pixel_data;
            }
            public int GetCorrect() {
                return Label;
            }
        }

        static void Main(string[] args) {
            Random rng = new Random();
            // numberDataTrainList のリストを作成
            List<NumberData> trainData = new List<NumberData>();
            // numberDataTrainList のリストを作成
            List<NumberData> testData = new List<NumberData>();

            // CSV ファイルの読み込み
            using (StreamReader reader = new StreamReader("mnist_train.csv")) {
                // CSV の各行を処理
                while (!reader.EndOfStream) {
                    string line = reader.ReadLine();
                    string[] values = line.Split(',');

                    if (values.Length == Constants.InputSize) {
                        try {
                            NumberData numberData = new NumberData(values);
                            trainData.Add(numberData);
                        }
                        catch (FormatException ex) {
                            Console.WriteLine($"データの変換エラー: {ex.Message}");
                        }
                        catch (ArgumentException ex) {
                            Console.WriteLine($"無視されたデータ: {ex.Message}");
                        }
                    }
                    else {
                        Console.WriteLine("不正なデータ行: " + line);
                    }
                }
            }

            using (StreamReader reader = new StreamReader("mnist_test.csv")) {
                // CSV の各行を処理
                while (!reader.EndOfStream) {
                    string line = reader.ReadLine();
                    string[] values = line.Split(',');

                    if (values.Length == Constants.InputSize) {
                        try {
                            NumberData numberData = new NumberData(values);
                            testData.Add(numberData);
                        }
                        catch (FormatException ex) {
                            Console.WriteLine($"データの変換エラー: {ex.Message}");
                        }
                        catch (ArgumentException ex) {
                            Console.WriteLine($"無視されたデータ: {ex.Message}");
                        }
                    }
                    else {
                        Console.WriteLine("不正なデータ行: " + line);
                    }
                }
            }


            // 重みを初期化 (3クラスに対して、それぞれ特徴量数分の重みを設定)
            List<double[]> weightsV = new List<double[]>();
            for (int i = 0; i < Constants.HiddenLayerSize; i++) {
                weightsV.Add(new double[Constants.InputSize]);  // Listに各行を追加
            }

            List<double[]> weightsW = new List<double[]>();
            for (int i = 0; i < Constants.OutputSize; i++) {
                weightsW.Add(new double[Constants.HiddenLayerSize + 1]);  // Listにクラスごとの重みを追加
            }

            for (int i = 0; i < weightsW.Count; i++) {
                for (int j = 0; j < weightsW[i].Length; j++) {
                    weightsW[i][j] = (rng.NextDouble() * 2.0 - 1.0) * Math.Sqrt(6.0 / (Constants.InputSize + Constants.HiddenLayerSize));  // 小さなランダム値で初期化
                }
            }
            for (int i = 0; i < weightsV.Count; i++) {
                for (int j = 0; j < weightsV[i].Length; j++) {
                    weightsV[i][j] = (rng.NextDouble() * 2.0 - 1.0) * Math.Sqrt(6.0 / (Constants.HiddenLayerSize + Constants.OutputSize));
                }
            }

            Console.WriteLine("学習開始");
            // 学習の実行
            TrainModel(trainData, weightsV, weightsW, Constants.LearningRate);
            Console.WriteLine("学習終了");

            // 最終的な重みの表示
            //Console.WriteLine("学習後の重み:");
            //for (int i = 0; i < weightsV.Count; i++) {
            //    Console.WriteLine($"v{i}の重み: {string.Join(", ", weightsV[i].Select(w => w.ToString("F2")))}");
            //}
            //Console.WriteLine();
            //for (int i = 0; i < weightsW.Count; i++) {
            //    Console.WriteLine($"w{i}の重み: {string.Join(", ", weightsW[i].Select(w => w.ToString("F2")))}");
            //}

            Console.WriteLine("検証開始");
            // 評価の実行
            EvaluateModel(testData, weightsW, weightsV);
        }

        // 学習用の関数 (トレーニングデータを使って重みを学習)
        static void TrainModel(List<NumberData> trainData, List<double[]> weightsV, List<double[]> weightsW, double learningRate) {
            var sw = new System.Diagnostics.Stopwatch();
            sw.Start();
            Random rng = new Random();
            double[] Losses = new double[Constants.Epochs]; // 損失の合計を記録
            for (int k = 0; k < Constants.Epochs; k++) {
                double totalLoss = 0.0; // 損失の合計を記録
                trainData = trainData.OrderBy(x => rng.Next()).ToList();

                if (k % 100 == 0) {
                    learningRate *= 0.5;
                }

                // Listに分割
                List<List<NumberData>> miniBatches = new List<List<NumberData>>();
                for (int i = 0; i < trainData.Count; i += Constants.BatchSize) {
                    List<NumberData> miniBatch = trainData.Skip(i).Take(Constants.BatchSize).ToList();
                    miniBatches.Add(miniBatch);
                }

                foreach (var miniBatch in miniBatches) {
                    double miniBatchLoss = 0.0; // ミニバッチの損失を記録
                    double[][] gradientsW = new double[weightsW.Count][];
                    double[][] gradientsV = new double[weightsV.Count][];

                    // 勾配の初期化
                    for (int i = 0; i < gradientsW.Length; i++) {
                        gradientsW[i] = new double[weightsW[i].Length];
                    }
                    for (int i = 0; i < gradientsV.Length; i++) {
                        gradientsV[i] = new double[weightsV[i].Length];
                    }

                    // ミニバッチ内のすべてのデータを処理
                    foreach (var numberData in miniBatch) {
                        double[] x = numberData.GetFeatures();
                        int yt = numberData.GetCorrect();

                        // 予測値を計算
                        double[] a = new double[weightsV.Count];
                        double[] b = new double[weightsV.Count + 1];
                        double[] u = new double[weightsW.Count];

                        // バイアス項の追加
                        b[0] = 1; // バイアス項
                        for (int i = 0; i < weightsV.Count; i++) {
                            a[i] = ScalarProduct(weightsV[i], x);
                            b[i + 1] = Sigmoid(a[i]);
                        }
                        for (int i = 0; i < weightsW.Count; i++) {
                            u[i] = ScalarProduct(weightsW[i], b);
                        }
                        double[] yp = SoftMax(u);

                        // 損失を計算してミニバッチの合計に追加
                        miniBatchLoss += CrossEntropyLoss(yp, yt);

                        // 出力層の誤差
                        double[] yd = new double[u.Length];
                        for (int i = 0; i < u.Length; i++) {
                            yd[i] = yp[i] - (yt == i ? 1 : 0); // 出力層の誤差
                        }

                        // 隠れ層の誤差
                        double[] bd = new double[b.Length];
                        for (int i = 0; i < b.Length; i++) {
                            double SumYdW = 0.0;
                            for (int j = 0; j < u.Length; j++) {
                                SumYdW += yd[j] * weightsW[j][i]; // 出力層の誤差を隠れ層へ伝播
                            }
                            bd[i] = b[i] * (1 - b[i]) * SumYdW; // シグモイドの勾配を考慮
                        }

                        // 勾配を蓄積
                        for (int i = 0; i < u.Length; i++) {
                            for (int j = 0; j < b.Length; j++) {
                                double gradientW = yd[i] * b[j]; // 出力層の誤差と隠れ層の出力の勾配
                                gradientsW[i][j] += gradientW; // 勾配を蓄積
                            }
                        }
                        for (int i = 0; i < a.Length; i++) {
                            for (int j = 0; j < x.Length; j++) {
                                double gradientV = bd[i] * x[j]; // 隠れ層の勾配と入力の積
                                gradientsV[i][j] += gradientV; // 勾配を蓄積
                            }
                        }
                    }

                    // ミニバッチのサイズで正規化して重みを更新
                    for (int i = 0; i < weightsW.Count; i++) {
                        for (int j = 0; j < weightsW[i].Length; j++) {
                            weightsW[i][j] -= (learningRate * gradientsW[i][j]) / miniBatch.Count;
                        }
                    }
                    for (int i = 0; i < weightsV.Count; i++) {
                        for (int j = 0; j < weightsV[i].Length; j++) {
                            weightsV[i][j] -= (learningRate * gradientsV[i][j]) / miniBatch.Count;
                        }
                    }

                    // ミニバッチ全体の損失を表示
                    double avgMiniBatchLoss = miniBatchLoss / miniBatch.Count;
                    //Console.WriteLine($"ミニバッチ損失: {avgMiniBatchLoss:F4}");
                    // エポック全体の損失を合計に追加
                    totalLoss += miniBatchLoss; // 追加
                }

                // 一周ごとの平均損失を表示
                double avgLoss = totalLoss / trainData.Count; // データの数で割る
                Console.WriteLine($"{k + 1}, 損失: {avgLoss:F4}");
                Console.WriteLine("1エポックにかかった時間");
                sw.Stop();
                TimeSpan ts = sw.Elapsed;
                Console.WriteLine($"　{ts.Hours}時間 {ts.Minutes}分 {ts.Seconds}秒 {ts.Milliseconds}ミリ秒");
                sw.Restart();
                Losses[k] = avgLoss;
            }

            WriteNumbersToFile(Losses);
        }

        // 検証用の関数 (テストデータを使ってモデルを評価)
        static void EvaluateModel(List<NumberData> testData, List<double[]> weightsW, List<double[]> weightsV) {
            int correctPredictions = 0;
            foreach (var irisData in testData) {
                double[] x = irisData.GetFeatures();

                // 予測を計算
                double[] a = new double[weightsV.Count];
                double[] b = new double[weightsV.Count + 1];
                double[] u = new double[weightsW.Count];
                //バイアス項の追加
                b[0] = 1;
                for (int i = 0; i < weightsV.Count; i++) {
                    a[i] = ScalarProduct(weightsV[i], x);
                    b[i + 1] = Sigmoid(a[i]);
                }
                for (int i = 0; i < weightsW.Count; i++) {
                    u[i] = ScalarProduct(weightsW[i], b);
                }
                double[] yp = SoftMax(u);

                // 予測ラベルを決定
                int predictedLabel = Array.IndexOf(yp, yp.Max());

                // 正解を確認
                if (predictedLabel == irisData.GetCorrect()) {
                    correctPredictions++;
                }
            }

            // 正解率を表示
            double accuracy = (double)correctPredictions / testData.Count;
            Console.WriteLine($"正解率: {accuracy * 100:F2}%");
        }

        // 内積を計算する関数
        static double ScalarProduct(double[] weights, double[] x) {
            double result = 0.0;
            for (int i = 0; i < x.Length; i++) {
                result += x[i] * weights[i];
            }
            return result;
        }

        // シグモイド関数
        static double Sigmoid(double x) {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        // SoftMax関数
        static double[] SoftMax(double[] u) {
            double maxU = u.Max();
            double sumExp = 0.0;
            for (int i = 0; i < u.Length; i++) {
                sumExp += Math.Exp(u[i] - maxU); // オーバーフロー対策
            }
            return u.Select(val => Math.Exp(val - maxU) / sumExp).ToArray();
        }

        // クロスエントロピー損失関数
        static double CrossEntropyLoss(double[] yp, int yt) {
            return -Math.Log(yp[yt]);
        }


        // リストを指定したサイズで分割するメソッド
        static List<List<T>> SplitList<T>(List<T> list, int batchSize) {
            List<List<T>> splitList = new List<List<T>>();
            for (int i = 0; i < list.Count; i += batchSize) {
                // 部分リストを作成して追加
                splitList.Add(list.GetRange(i, Math.Min(batchSize, list.Count - i)));
            }
            return splitList;
        }

        static void WriteNumbersToFile(double[] numbers) {
            string filePath = Path.Combine("..", "..", "..", "..", "data.txt");
            bool fileExists = File.Exists(filePath); // ファイルの存在を確認
            string title = "計算結果";
            using (StreamWriter writer = new StreamWriter(filePath)) // 追記モード
            {
                // ファイルが存在しない場合のみタイトルを書き込む
                if (!fileExists) {
                    writer.WriteLine(title);
                }

                // 配列の数字を書き込む
                foreach (double number in numbers) {
                    writer.WriteLine(number);
                }

                Console.WriteLine("結果をファイルに書き込みました: " + filePath);
            }

        }
    }
}
