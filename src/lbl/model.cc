#include "lbl/model.h"

#include <iomanip>

#include <boost/make_shared.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "lbl/context_cache.h"
#include "lbl/minibatch_words.h"
#include "lbl/factored_metadata.h"
#include "lbl/factored_maxent_metadata.h"
#include "lbl/factored_weights.h"
#include "lbl/global_factored_maxent_weights.h"
#include "lbl/metadata.h"
#include "lbl/minibatch_factored_maxent_weights.h"
#include "lbl/model_utils.h"
#include "lbl/operators.h"
#include "lbl/weights.h"
#include "utils/conditional_omp.h"

namespace oxlm {

template<class GlobalWeights, class MinibatchWeights, class Metadata>
Model<GlobalWeights, MinibatchWeights, Metadata>::Model() {}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
Model<GlobalWeights, MinibatchWeights, Metadata>::Model(
    const boost::shared_ptr<ModelData>& config)
    : config(config) {
  metadata = boost::make_shared<Metadata>(config, vocab);
  srand(1);
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
boost::shared_ptr<Vocabulary> Model<GlobalWeights, MinibatchWeights, Metadata>::getVocab() const {
  return vocab;
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
boost::shared_ptr<ModelData> Model<GlobalWeights, MinibatchWeights, Metadata>::getConfig() const {
  return config;
}


template<class GlobalWeights, class MinibatchWeights, class Metadata>
MatrixReal Model<GlobalWeights, MinibatchWeights, Metadata>::getWordVectors() const {
  return weights->getWordVectors();
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
VectorReal Model<GlobalWeights, MinibatchWeights, Metadata>::getWordBias() const {
  return weights->getWordBias();
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
ContextTransformsType Model<GlobalWeights, MinibatchWeights, Metadata>::getTransformationMatrix() const {
  return weights->getTransformationMatrix();
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
MatrixReal Model<GlobalWeights, MinibatchWeights, Metadata>::getWordContextVectors() const {
  return weights->getWordContextVectors();
}


template<class GlobalWeights, class MinibatchWeights, class Metadata>
WeightsType Model<GlobalWeights, MinibatchWeights, Metadata>::getW() const {
  return weights->getW();
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
Real* Model<GlobalWeights, MinibatchWeights, Metadata>::getdata() const {
  return weights->data;
}
  
template<class GlobalWeights, class MinibatchWeights, class Metadata>
void Model<GlobalWeights, MinibatchWeights, Metadata>::learn() {
  // Initialize the vocabulary now, if it hasn't been initialized when the
  // vocabulary was partitioned in classes.
  boost::shared_ptr<Corpus> training_corpus =
      readTrainingCorpus(config, vocab);

  boost::shared_ptr<Corpus> test_corpus;
  if (config->test_file.size()) {
    test_corpus = readTestCorpus(config, vocab);
  }

  if (config->model_input_file.size() == 0) {
    // Train a new model.
    metadata->initialize(training_corpus);
    weights = boost::make_shared<GlobalWeights>(
        config, metadata, training_corpus);
    weights->printInfo();
  } else {
    // Continue training an existing model.
    Real log_likelihood = 0;
    evaluate(test_corpus, log_likelihood);
    cout << "Initial perplexity: "
         << perplexity(log_likelihood, test_corpus->size()) << endl;
  }

  vector<int> indices(training_corpus->size());
  iota(indices.begin(), indices.end(), 0);

  int best_minibatch = 0;
  Real best_perplexity = numeric_limits<Real>::infinity();
  Real global_objective = 0, test_objective = 0;
  boost::shared_ptr<MinibatchWeights> global_gradient =
      boost::make_shared<MinibatchWeights>(config, metadata);
  boost::shared_ptr<GlobalWeights> adagrad =
      boost::make_shared<GlobalWeights>(config, metadata);
  MinibatchWords global_words;

  int shared_index = 0;
  // For no particular reason. It just looks like this works best.
  int task_size = sqrt(config->minibatch_size);

  omp_set_num_threads(config->threads);
  #pragma omp parallel
  {
    int minibatch_counter = 1;
    int minibatch_size = config->minibatch_size;
    int minibatch_threshold = config->minibatch_threshold;
    boost::shared_ptr<MinibatchWeights> gradient =
        boost::make_shared<MinibatchWeights>(config, metadata);

    int iter = 0;
    while (iter < config->iterations &&
           minibatch_counter - best_minibatch <= minibatch_threshold) {
      auto iteration_start = GetTime();

      #pragma omp master
      {
        if (config->randomise) {
          random_shuffle(indices.begin(), indices.end());
        }
        global_objective = 0;
      }
      // Wait until the master thread finishes shuffling the indices.
      #pragma omp barrier

      size_t start = 0;
      while (start < training_corpus->size() &&
             minibatch_counter - best_minibatch <= minibatch_threshold) {
        size_t end = min(training_corpus->size(), start + minibatch_size);

        vector<int> minibatch(
            indices.begin() + start,
            min(indices.begin() + end, indices.end()));
        global_gradient->init(training_corpus, minibatch);
        // Reset the set of minibatch words shared across all threads.
        #pragma omp master
        {
          global_words = MinibatchWords();
          shared_index = 0;
        }

        gradient->init(training_corpus, minibatch);

        // Wait until the global gradient is initialized. Otherwise, some
        // gradient updates may be ignored.
        #pragma omp barrier

        Real objective = 0;
        MinibatchWords words;
        size_t task_start;
        while (true) {
          #pragma omp critical
          {
            task_start = shared_index;
            shared_index += task_size;
          }

          if (task_start < minibatch.size()) {
            size_t task_end = min(task_start + task_size, minibatch.size());
            vector<int> task(
                minibatch.begin() + task_start, minibatch.begin() + task_end);
            if (config->noise_samples > 0) {
              weights->estimateGradient(
                  training_corpus, task, gradient, objective, words);
            } else {
              weights->getGradient(
                  training_corpus, task, gradient, objective, words);
            }
          } else {
            break;
          }
        }

        global_gradient->syncUpdate(words, gradient);
        #pragma omp critical
        {
          global_objective += objective;
          global_words.merge(words);
        }

        // Wait until the global gradient is fully updated by all threads and
        // the global words are fully merged.
        #pragma omp barrier

        // Prepare minibatch words for parallel processing.
        #pragma omp master
        global_words.transform();

        // Wait until the minibatch words are fully prepared for parallel
        // processing.
        #pragma omp barrier

        update(global_words, global_gradient, adagrad);

        // Wait for all threads to finish making the model gradient update.
        #pragma omp barrier

        Real minibatch_factor =
            static_cast<Real>(end - start) / training_corpus->size();
        objective = regularize(global_gradient, minibatch_factor);
        #pragma omp critical
        global_objective += objective;

        // Clear gradients.
        gradient->clear(words, false);
        global_gradient->clear(global_words, true);

        // Wait the regularization update to finish and make sure the global
        // words are reset only after the global gradient is fully cleared.
        #pragma omp barrier

        if (minibatch_counter % config->evaluate_frequency == 0) {
          evaluate(test_corpus, iteration_start, minibatch_counter,
                   test_objective, best_perplexity, best_minibatch);
        }

        ++minibatch_counter;
        start = end;
      }

      evaluate(test_corpus, iteration_start, minibatch_counter,
               test_objective, best_perplexity, best_minibatch);
      #pragma omp master
      {
        Real iteration_time = GetDuration(iteration_start, GetTime());
        cout << "Iteration: " << iter << ", "
             << "Time: " << iteration_time << " seconds, "
             << "Objective: " << global_objective / training_corpus->size()
             << endl;
        cout << endl;
      }

      ++iter;
    }
  }

  cout << "Overall minimum perplexity: " << best_perplexity << endl;
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
void Model<GlobalWeights, MinibatchWeights, Metadata>::update(
    const MinibatchWords& global_words,
    const boost::shared_ptr<MinibatchWeights>& global_gradient,
    const boost::shared_ptr<GlobalWeights>& adagrad) {
  adagrad->updateSquared(global_words, global_gradient);
  weights->updateAdaGrad(global_words, global_gradient, adagrad);
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
Real Model<GlobalWeights, MinibatchWeights, Metadata>::regularize(
    const boost::shared_ptr<MinibatchWeights>& global_gradient,
    Real minibatch_factor) {
  return weights->regularizerUpdate(global_gradient, minibatch_factor);
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
void Model<GlobalWeights, MinibatchWeights, Metadata>::evaluate(
    const boost::shared_ptr<Corpus>& test_corpus, Real& accumulator) const {
  if (test_corpus != nullptr) {
    #pragma omp master
    {
      cout << "Calculating perplexity for " << test_corpus->size()
           << " tokens..." << endl;
      accumulator = 0;
    }

    // Each thread must wait until the perplexity is set to 0.
    // Otherwise, partial results might get overwritten.
    #pragma omp barrier

    vector<int> indices(test_corpus->size());
    iota(indices.begin(), indices.end(), 0);
    size_t start = 0;
    int minibatch_size = sqrt(config->minibatch_size);
    while (start < test_corpus->size()) {
      size_t end = min(start + minibatch_size, test_corpus->size());
      vector<int> minibatch(
          indices.begin() + start, min(indices.begin() + end, indices.end()));
      minibatch = scatterMinibatch(minibatch);

      Real log_likelihood = weights->getLogLikelihood(test_corpus, minibatch);
      #pragma omp critical
      accumulator += log_likelihood;

      start = end;
    }

    // Wait for all the threads to compute the perplexity for their slice of
    // test data.
    #pragma omp barrier
  }
}


template<class GlobalWeights, class MinibatchWeights, class Metadata>
void Model<GlobalWeights, MinibatchWeights, Metadata>::evaluate(
    const boost::shared_ptr<Corpus>& test_corpus, const Time& iteration_start,
    int minibatch_counter, Real& log_likelihood,
    Real& best_perplexity, int& best_minibatch) const {
  if (test_corpus != nullptr) {
    evaluate(test_corpus, log_likelihood);

    #pragma omp master
    {
      Real test_perplexity = perplexity(log_likelihood, test_corpus->size());
      Real iteration_time = GetDuration(iteration_start, GetTime());
      cout << "\tMinibatch " << minibatch_counter << ", "
           << "Time: " << GetDuration(iteration_start, GetTime()) << " seconds, "
           << "Test Perplexity: " << test_perplexity << endl;

      if (test_perplexity < best_perplexity) {
        best_perplexity = test_perplexity;
        best_minibatch = minibatch_counter;
        save();
      }
    }
  } else {
    #pragma omp master
    save();
  }
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
Real Model<GlobalWeights, MinibatchWeights, Metadata>::getLogProb(
    int word_id, const vector<int>& context) const {
  return weights->getLogProb(word_id, context);
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
Real Model<GlobalWeights, MinibatchWeights, Metadata>::getUnnormalizedScore(
    int word_id, const vector<int>& context) const {
  return weights->getUnnormalizedScore(word_id, context);
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
void Model<GlobalWeights, MinibatchWeights, Metadata>::save() const {
  if (config->model_output_file.size()) {
    cout << "Writing model to " << config->model_output_file << "..." << endl;
    ofstream fout(config->model_output_file);
    boost::archive::binary_oarchive oar(fout);
    oar << config;
    oar << vocab;
    oar << weights;
    oar << metadata;
    cout << "Done..." << endl;
  }
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
void Model<GlobalWeights, MinibatchWeights, Metadata>::load(const string& filename) {
  if (filename.size() > 0) {
    auto start_time = GetTime();
    cerr << "Loading model from " << filename << "..." << endl;
    ifstream fin(filename);
    boost::archive::binary_iarchive iar(fin);
    iar >> config;
    iar >> vocab;
    iar >> weights;
    iar >> metadata;
    cerr << "Reading model took " << GetDuration(start_time, GetTime())
         << " seconds..." << endl;
  }
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
void Model<GlobalWeights, MinibatchWeights, Metadata>::clearCache() {
  weights->clearCache();
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
bool Model<GlobalWeights, MinibatchWeights, Metadata>::operator==(
    const Model<GlobalWeights, MinibatchWeights, Metadata>& other) const {
  return *config == *other.config
      && *metadata == *other.metadata
      && *weights == *other.weights;
}

template class Model<Weights, Weights, Metadata>;
template class Model<FactoredWeights, FactoredWeights, FactoredMetadata>;
template class Model<GlobalFactoredMaxentWeights, MinibatchFactoredMaxentWeights, FactoredMaxentMetadata>;
template class Model<SourceFactoredWeights, SourceFactoredWeights, FactoredMetadata>;
template class Model<FactoredTreeWeights, FactoredTreeWeights, TreeMetadata>;

} // namespace oxlm
