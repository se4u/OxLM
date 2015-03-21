#pragma once

#include <boost/shared_ptr.hpp>

#include "corpus/corpus.h"
#include "lbl/config.h"
#include "lbl/factored_metadata.h"
#include "lbl/factored_maxent_metadata.h"
#include "lbl/factored_weights.h"
#include "lbl/factored_tree_weights.h"
#include "lbl/global_factored_maxent_weights.h"
#include "lbl/metadata.h"
#include "lbl/minibatch_factored_maxent_weights.h"
#include "lbl/minibatch_words.h"
#include "lbl/model_utils.h"
#include "lbl/parallel_vocabulary.h"
#include "lbl/source_factored_weights.h"
#include "lbl/tree_metadata.h"
#include "lbl/utils.h"
#include "lbl/vocabulary.h"
#include "lbl/weights.h"
#include "lbl/parallel_corpus.h"

namespace oxlm {

enum ModelType {
  NLM = 1,
  FACTORED_NLM = 2,
  FACTORED_MAXENT_NLM = 3,
  SOURCE_FACTORED_NLM = 4,
  FACTORED_TREE_NLM = 5,
};

template<class GlobalWeights, class MinibatchWeights, class Metadata>
class Model {
 public:
  Model();

  Model(const boost::shared_ptr<ModelData>& config);

  boost::shared_ptr<Vocabulary> getVocab() const;

  boost::shared_ptr<ModelData> getConfig() const;

  void learn();

  void update(
      const MinibatchWords& global_words,
      const boost::shared_ptr<MinibatchWeights>& global_gradient,
      const boost::shared_ptr<GlobalWeights>& adagrad);

  Real regularize(
      const boost::shared_ptr<MinibatchWeights>& global_gradient,
      Real minibatch_factor);

  void evaluate(
      const boost::shared_ptr<Corpus>& corpus, Real& accumulator) const;

  Real getLogProb(int word_id, const vector<int>& context) const;

  Real getUnnormalizedScore(int word_id, const vector<int>& context) const;

  MatrixReal getWordVectors() const;
  VectorReal getWordBias() const;
  ContextTransformsType getTransformationMatrix() const;
  MatrixReal getWordContextVectors() const;
  
  WeightsType getW() const;
  Real* getdata() const;
  
  void save() const;

  void load(const string& filename);

  void clearCache();

  bool operator==(
      const Model<GlobalWeights, MinibatchWeights, Metadata>& other) const;
  boost::shared_ptr<GlobalWeights> weights;
 private:
  void evaluate(
      const boost::shared_ptr<Corpus>& corpus, const Time& iteration_start,
      int minibatch_counter, Real& objective,
      Real& best_perplexity, int& best_minibatch) const;

  boost::shared_ptr<ModelData> config;
  boost::shared_ptr<Vocabulary> vocab;
  boost::shared_ptr<Metadata> metadata;

};

class LM : public Model<Weights, Weights, Metadata> {
 public:
  LM() : Model<Weights, Weights, Metadata>() {}

  LM(const boost::shared_ptr<ModelData>& config)
      : Model<Weights, Weights, Metadata>(config) {}
};

class FactoredLM: public Model<FactoredWeights, FactoredWeights, FactoredMetadata> {
 public:
  FactoredLM() : Model<FactoredWeights, FactoredWeights, FactoredMetadata>() {}

  FactoredLM(const boost::shared_ptr<ModelData>& config)
      : Model<FactoredWeights, FactoredWeights, FactoredMetadata>(config) {}
};

class FactoredMaxentLM : public Model<GlobalFactoredMaxentWeights, MinibatchFactoredMaxentWeights, FactoredMaxentMetadata> {
 public:
  FactoredMaxentLM() : Model<GlobalFactoredMaxentWeights, MinibatchFactoredMaxentWeights, FactoredMaxentMetadata>() {}

  FactoredMaxentLM(const boost::shared_ptr<ModelData>& config)
      : Model<GlobalFactoredMaxentWeights, MinibatchFactoredMaxentWeights, FactoredMaxentMetadata>(config) {}
};

class SourceFactoredLM: public Model<SourceFactoredWeights, SourceFactoredWeights, FactoredMetadata> {
 public:
  SourceFactoredLM() : Model<SourceFactoredWeights, SourceFactoredWeights, FactoredMetadata>() {}

  SourceFactoredLM(const boost::shared_ptr<ModelData>& config)
      : Model<SourceFactoredWeights, SourceFactoredWeights, FactoredMetadata>(config) {}
};

class FactoredTreeLM : public Model<FactoredTreeWeights, FactoredTreeWeights, TreeMetadata> {
 public:
  FactoredTreeLM() : Model<FactoredTreeWeights, FactoredTreeWeights, TreeMetadata>() {}

  FactoredTreeLM(const boost::shared_ptr<ModelData>& config)
      : Model<FactoredTreeWeights, FactoredTreeWeights, TreeMetadata>(config) {}
};

} // namespace oxlm
