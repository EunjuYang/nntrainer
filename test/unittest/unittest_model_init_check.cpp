#include <gtest/gtest.h>
#include <model.h>
#include <layer.h>

TEST(TMP_Model_Test, fc) {
  auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model->setProperty({"batch_size=1", "epochs=1"});

  std::shared_ptr<ml::train::Layer> input_layer1 = ml::train::createLayer("input", {"name=in1", "input_shape=40:1:1024"});
  std::shared_ptr<ml::train::Layer> fc = ml::train::createLayer("fully_connected", {"unit=1", "input_layers=[in1]"});
  model->addLayer(fc);

  model->compile();
  model->initialize();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}