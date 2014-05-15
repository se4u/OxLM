#pragma once

#include <boost/serialization/shared_ptr.hpp>

#include "lbl/archive_export.h"
#include "lbl/bloom_filter.h"
#include "lbl/feature_context_keyer.h"
#include "lbl/feature_filter.h"
#include "lbl/ngram_query.h"

namespace oxlm {

class FeatureApproximateFilter : public FeatureFilter {
 public:
  FeatureApproximateFilter();

  FeatureApproximateFilter(
      int num_candidates, const boost::shared_ptr<FeatureContextKeyer>& keyer,
      const boost::shared_ptr<BloomFilter<NGramQuery>>& bloom_filter);

  virtual vector<int> getIndexes(const FeatureContext& feature_context) const;

  bool operator==(const FeatureApproximateFilter& other) const;

  virtual ~FeatureApproximateFilter();

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & boost::serialization::base_object<FeatureFilter>(*this);
    ar & numCandidates;
    ar & keyer;
    ar & bloomFilter;
  }

  int numCandidates;
  boost::shared_ptr<FeatureContextKeyer> keyer;
  boost::shared_ptr<BloomFilter<NGramQuery>> bloomFilter;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::FeatureApproximateFilter)
