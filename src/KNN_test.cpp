template <typename T>
void test(T& model, std::string dataset, int k) {

  auto start = std::chrono::system_clock::now();

  std::string data_name = "../data/" + dataset + "/data.csv";

  data complete_data = load(data_name);
  std::random_shuffle(complete_data.begin(), complete_data.end());

  // Optional data scaling
  complete_data = scale(complete_data);

  data training_data;
  data test_data;

  int stride = floor(complete_data.size()/30);

  for (int j=0; j<complete_data.size(); j++) {
    if (j < stride) {
      test_data.push_back(complete_data[j]);
    } else {
      training_data.push_back(complete_data[j]);
    }
  }

  model.train(training_data);

  double count = 0;
  for (int i=0; i<test_data.size(); i++) {
    std::vector<double> x(test_data[i].begin(), test_data[i].end()-1);
    double y = test_data[i].back();
    double a = model.h(x, k);
    if (a == y) {
      count++;
    }
  }

  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = stop-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(stop);

  // Report
  std::cout << "Date: " << std::ctime(&end_time)<< '\n';
  std::cout << "Dataset: " << dataset << '\n';
  std::cout << "Model: KNN" << '\n';
  std::cout << "Scaling: Yes" << '\n';
  std::cout << "K: " << k << '\n';
  std::cout << "Final test classification accuracy: " << 100*count/test_data.size() << '%' << '\n';
  std::cout << "Time running: " << elapsed_seconds.count() << " seconds" << '\n';

  std::cout << "--------------------------------------------------------------------------------" << '\n';

}
