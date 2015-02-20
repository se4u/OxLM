#include <boost/program_options.hpp>

#include "lbl/metadata.h"
#include "lbl/model.h"
#include "lbl/weights.h"
#include "utils/git_revision.h"

using namespace boost::program_options;
using namespace oxlm;

template<class Model>
void printout(const string& model_file){
  Model model;
  model.load(model_file);
  // Target vector // R.m_storage.m_data[0]
  /* print data[100*10003]@10
$19 = {0.0518091694, 0.023289796, 0.0298359208, -0.0147829736, 0.052163478, 
  -0.00912516098, -0.00602822192, -0.0731897056, -0.00670113787, -0.0174017884}
(gdb) print R.m_storage.m_data[0]@10
$20 = {0.0518091694, 0.023289796, 0.0298359208, -0.0147829736, 0.052163478, 
  -0.00912516098, -0.00602822192, -0.0731897056, -0.00670113787, -0.0174017884}
  */
  auto R = model.getWordVectors(); 
  WeightsType W = model.getW();
  Real* data = model.getdata();
  ofstream fout("oxmodel.txt");
  // Unfortuantely just printing out this stuff would be useless
  // But its a start.
  // (100*10003+100*10003+100+10003)==W.size()
  // This means that I am storing Q, then R, then 1 less than the context width, and then no H, and then B.
  cout << W.size() << endl;
  for(int i=0; i<W.size(); i++){
    cout << data[i] << endl;
  };
  cout << "Done..." << endl;
  // boost::shared_ptr<ModelData> config = model.getConfig();
  // boost::shared_ptr<Vocabulary> vocab = model.getVocab();
  
  // auto C = model.getWordVectors(); // Transformation
  // auto Q = model.getWordVectors(); // Context embedding
  // auto B = model.getWordVectors(); // Bias vector
  
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

      
