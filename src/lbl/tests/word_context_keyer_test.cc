#include "gtest/gtest.h"

#include <boost/make_shared.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "lbl/word_context_keyer.h"

namespace ar = boost::archive;

namespace oxlm {

TEST(WordContextKeyerTest, TestBasic) {
  WordContextKeyer keyer(13, 100);
  vector<int> context = {1};
  EXPECT_EQ(97, keyer.getKey(context));
  context = {1, 2};
  EXPECT_EQ(17, keyer.getKey(context));
  context = {1, 2, 3};
  EXPECT_EQ(32, keyer.getKey(context));
}

TEST(WordContextKeyerTest, TestSerialization) {
  boost::shared_ptr<FeatureContextKeyer> keyer_ptr =
      boost::make_shared<WordContextKeyer>(13, 100);

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << keyer_ptr;

  boost::shared_ptr<FeatureContextKeyer> keyer_copy_ptr;
  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> keyer_copy_ptr;

  boost::shared_ptr<WordContextKeyer> expected_ptr =
      dynamic_pointer_cast<WordContextKeyer>(keyer_ptr);
  boost::shared_ptr<WordContextKeyer> actual_ptr =
      dynamic_pointer_cast<WordContextKeyer>(keyer_copy_ptr);

  EXPECT_NE(nullptr, expected_ptr);
  EXPECT_NE(nullptr, actual_ptr);
  EXPECT_EQ(*expected_ptr, *actual_ptr);
}

} // namespace oxlm