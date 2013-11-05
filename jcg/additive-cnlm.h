#ifndef OXLM_JCG_ADDITIVE_CNLM_H
#define OXLM_JCG_ADDITIVE_CNLM_H

#include <boost/shared_ptr.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <iostream>
#include <fstream>
#include <vector>

#include <Eigen/Dense>

#include "corpus/corpus.h"
#include "cg/config.h"
#include "cg/utils.h"

#include "jcg/gcnlm.h"

namespace oxlm {

class ConditionalNLM : public GeneralConditionalNLM {
public:
  ConditionalNLM();
  ConditionalNLM(const ModelData& config, const Dict& source_vocab, const Dict& target_vocab, const std::vector<int>& classes);
  ~ConditionalNLM() { delete [] a_data; }

  int source_types() const { return m_source_labels.size(); }

  const Dict& source_label_set() const { return m_source_labels; }
  Dict& source_label_set() { return m_source_labels; }

  // Better name needed. Make source_corpus const again.
  Real gradient(std::vector<Sentence>& source_corpus, const std::vector<Sentence>& target_corpus,
                const TrainingInstances &training_instances, Real l2, Real source_l2, WeightsType& g_W);

  Real log_prob(const WordId w, const std::vector<WordId>& context, const Sentence& source, bool cache=false, int target_index=-1) const;

  /*virtual*/int num_weights() const { return m_data_size + a_data_size; }

  MatrixReal window_product(int i, const MatrixReal& v, bool transpose=false) const {
    if (config.diagonal)
      return (T.at(i).asDiagonal() * v.transpose()).transpose();
    else if (transpose) return v * T.at(i).transpose();
    else                return v * T.at(i);
  }

  /*virtual*/void source_repr_callback(TrainingInstance t, int t_i, VectorReal& r);
  /*virtual*/void source_grad_callback(TrainingInstance t, int t_i, int instance_counter, const VectorReal& grads);
  void source_representation(const Sentence& source, int target_index, VectorReal& result) const;

  ContextTransformsType T;  // source window context transforms
  WordVectorsType       S;  // source word representations
  WeightsType           WA; // All the parameters in one vector

  ContextTransformsType g_T;  // source window context transforms
  WordVectorsType       g_S;  // source word representations

  std::vector<Sentence> source_corpus;

protected:
  /*virtual*/void init(bool init_weights=false);
  /*virtual*/void allocate_data();
  void map_parameters(WeightsType& wa, WordVectorsType& s,
                      ContextTransformsType& t) const;

  Dict m_source_labels;
  int a_data_size;
  Real* a_data;

public:
  friend class boost::serialization::access;
  template<class Archive>
  void save(Archive & ar, const unsigned int version) const {
    ar << config;
    ar << m_target_labels;
    ar << m_source_labels;
    ar << boost::serialization::make_array(m_data, m_data_size);

    ar << word_to_class;
    ar << indexes;
    ar << length_ratio;
  }

  template<class Archive>
  void load(Archive & ar, const unsigned int version) {
    ar >> config;
    ar >> m_target_labels;
    ar >> m_source_labels;
    delete [] m_data;
    delete [] a_data;
    // TODO(kmh): Clean up archiving
    init(false);
    ar >> boost::serialization::make_array(m_data, m_data_size);

    ar >> word_to_class;
    ar >> indexes;
    ar >> length_ratio;
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER();
};

typedef std::shared_ptr<ConditionalNLM> ConditionalNLMPtr;

}  // namespace oxlm
#endif  // OXLM_JCG_ADDITIVE_CNLM_H
