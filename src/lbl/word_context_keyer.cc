#include "lbl/word_context_keyer.h"

namespace oxlm {

WordContextKeyer::WordContextKeyer() {}

WordContextKeyer::WordContextKeyer(
    int class_id, int num_words, int hash_space_size)
    : classId(class_id), numWords(num_words), hashSpaceSize(hash_space_size) {}

int WordContextKeyer::getKey(const FeatureContext& feature_context) const {
  NGramQuery query(numWords + classId, feature_context.data);
  return hash_function(query) % hashSpaceSize;
}

NGramQuery WordContextKeyer::getPrediction(
    int candidate, const FeatureContext& feature_context) const {
  vector<int> context(1, numWords + classId);
  context.insert(
      context.end(), feature_context.data.begin(), feature_context.data.end());
  return NGramQuery(candidate, context);
}

bool WordContextKeyer::operator==(const WordContextKeyer& other) const {
  return classId == other.classId
      && numWords == other.numWords
      && hashSpaceSize == other.hashSpaceSize;
}

WordContextKeyer::~WordContextKeyer() {}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::WordContextKeyer)
