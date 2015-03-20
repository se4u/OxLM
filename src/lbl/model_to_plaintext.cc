#include <boost/program_options.hpp>
#include <stdlib.h>
#include "lbl/metadata.h"
#include "lbl/model.h"
#include "lbl/weights.h"
#include "lbl/utils.h"
#include "utils/git_revision.h"

using namespace boost::program_options;
using namespace oxlm;

VectorReal my_prediction_vector(const vector<int>& context,
			       const MatrixReal Q,
			       const ContextTransformsType C,
			       const boost::shared_ptr<ModelData> config
			       ){
  VectorReal prediction_vector = VectorReal::Zero(Q.rows());
  for(auto i = 0; i != context.size(); i++) {
    auto a = C[i].asDiagonal();
    auto b = Q.col(context[i]);
    prediction_vector +=  a*b ;
  }
  prediction_vector = activation(config, prediction_vector);
  assert(config->hidden_layers == 0);
  return prediction_vector;
}

Real my_unnormalized_score(int word_id,
	       vector<int> context,
	      const MatrixReal R,
	      const MatrixReal Q,
	      const ContextTransformsType C,
	      const VectorReal B,
	      boost::shared_ptr<ModelData>& config){
  VectorReal prediction_vector = my_prediction_vector(context,
						     Q,
						     C,
						     config);
  auto ret = R.col(word_id).dot(prediction_vector)+B(word_id);
  return ret;
}

Real my_log_prob(int word_id,
		 vector<int>& context,
		 const MatrixReal& R,
		 const MatrixReal& Q,
		 const ContextTransformsType& C,
		 const VectorReal& B,
		 boost::shared_ptr<ModelData>& config
		 )  {
  VectorReal prediction_vector = my_prediction_vector(context, Q, C, config);
  VectorReal word_probs = oxlm::logSoftMax(R.transpose() * prediction_vector + B);
  return word_probs(word_id);
}


void print_matrix(Eigen::Matrix<float, -1, -1> M,
		  std::string start_str,
		  std::string end_str){
cout << endl << start_str << endl;
  for(int i= 0; i<M.rows(); i++){
    for(int j = 0; j<M.cols(); j++){
	cout << M(i, j) << ' ';
    }
    cout << endl;
  }
  cout << end_str << endl;
}

void print_contexttransform(ContextTransformsType C,
			    int width ,
			    std::string start_str,
			    std::string end_str){
  cout << endl << start_str << endl;
  for (int i=0; i<C[0].rows(); i++){
    for (int j = 0; j<width; j++){
      cout << C[j](i) << ' ';
    }
    cout << endl;
  }
  cout << end_str << endl;
}

template<class Model>
void printout(const string& model_file){
  Model model;
  model.load(model_file);
  boost::shared_ptr<ModelData> config = model.getConfig();
  // cout << config->iterations << endl;
  // cout << config->ngram_order << endl;
  // cout << config->diagonal_contexts << endl;
  // cout << config->hidden_layers << endl;
  
  //  Print the vocabulary
  boost::shared_ptr<Vocabulary> vocab = model.getVocab();
  auto dic = (vocab->dict);
  std::map<std::string, int> d = dic.d_;
  cout << endl << "VOCABULARY_START" << endl;
  for(auto it = d.begin(); it != d.end(); it++){
    cout << it->first << ' ' << it->second << endl;
  }
  cout << "VOCABULARY_END" << endl;
  // Now print parameters
  MatrixReal R = model.getWordVectors(); // The target word embeddings
  MatrixReal Q = model.getWordContextVectors(); // Context embedding
  ContextTransformsType C = model.getTransformationMatrix(); // Transformation
  VectorReal B = model.getWordBias(); // Bias vector
  print_matrix(R.transpose(), "R_START", "R_END");
  print_matrix(Q.transpose(), "Q_START", "Q_END");
  print_matrix(B, "B_START", "B_END");
  print_contexttransform(C, config->ngram_order - 1,"C_START", "C_END");
  // TESTING CODE (all tests passed)
  // int word_id = 4;
  // static const int arr[] = {16};
  // vector<int> context (arr, arr + sizeof(arr) / sizeof(arr[0]) );
  // Real oxlm_score = model.getUnnormalizedScore(word_id, context);
  // Real my_score = my_unnormalized_score(word_id,
  // 					context,
  // 					R,
  // 					Q,
  // 					C,
  // 					B,
  // 					config);
  // assert(abs(oxlm_score - my_score) < 1e-10);
  
  
}
int main(int argc, char** argv) {
  // First create a structure to hold options
  options_description desc("Command line options");
  desc.add_options()
    ("help,h", "Print help message.")
    ("model,m", value<string>(),
       "Model file to convert to plain text")
    ("type,t", value<int>()->required(), "Model type");

  // Convert the options given to a map
  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 0;
  }
  notify(vm);
  
  string model_file = vm["model"].as<string>();
  ModelType model_type = static_cast<ModelType>(vm["type"].as<int>());

  switch (model_type){
  case NLM:
      printout<LM>(model_file);
      return 0;
  default:
    cout << "Unknown model type" << endl;
    return 1;
  }
  return 0;
}

      
