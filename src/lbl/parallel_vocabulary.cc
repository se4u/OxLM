#include "lbl/parallel_vocabulary.h"

namespace oxlm {

int ParallelVocabulary::convertSource(const string& word, bool frozen) {
  return sourceDict.Convert(word, frozen);
}

} // namespace oxlm