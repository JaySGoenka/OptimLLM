const fs = require("fs");
const path = require("path");

const PROJECT_ROOT = path.join(__dirname, "..");
const TRAINING_PATH = path.join(PROJECT_ROOT, "data", "router-training.json");
const MODEL_PATH = path.join(PROJECT_ROOT, "data", "router-model.json");
const TARGETS = ["task_type", "difficulty", "privacy", "route_class"];
const MIN_TOKEN_COUNT = 1;

const STOP_WORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
  "is", "it", "me", "my", "of", "on", "or", "that", "the", "this", "to", "with"
]);

function main() {
  const trainingData = JSON.parse(fs.readFileSync(TRAINING_PATH, "utf8"));
  validateTrainingData(trainingData);

  const examples = trainingData.examples.map((example) => ({
    ...example,
    tokens: tokenize(example.text)
  }));
  const vocabulary = buildVocabulary(examples);
  const classifiers = {};

  for (const target of TARGETS) {
    classifiers[target] = trainNaiveBayes(examples, vocabulary, target, trainingData.labels[target]);
  }

  const model = {
    schema_version: 1,
    model_type: "multinomial_naive_bayes",
    trained_at: new Date().toISOString(),
    training_examples: examples.length,
    vocabulary,
    targets: TARGETS,
    classifiers,
    metrics: evaluateTrainingAccuracy(examples, classifiers)
  };

  fs.writeFileSync(MODEL_PATH, `${JSON.stringify(model, null, 2)}\n`);
  printSummary(model);
}

function validateTrainingData(trainingData) {
  if (!Array.isArray(trainingData.examples) || trainingData.examples.length === 0) {
    throw new Error("router-training.json must include non-empty examples.");
  }

  for (const target of TARGETS) {
    if (!Array.isArray(trainingData.labels?.[target])) {
      throw new Error(`router-training.json is missing labels.${target}.`);
    }
  }

  for (const [index, example] of trainingData.examples.entries()) {
    if (!example.text || typeof example.text !== "string") {
      throw new Error(`Example ${index} is missing text.`);
    }

    for (const target of TARGETS) {
      if (!trainingData.labels[target].includes(example[target])) {
        throw new Error(`Example ${index} has invalid ${target}: ${example[target]}`);
      }
    }
  }
}

function tokenize(text) {
  const normalized = text
    .toLowerCase()
    .replace(/[^a-z0-9+#.]+/g, " ")
    .trim();

  if (!normalized) {
    return [];
  }

  const words = normalized
    .split(/\s+/)
    .filter((word) => word.length > 1 && !STOP_WORDS.has(word));
  const bigrams = [];

  for (let index = 0; index < words.length - 1; index += 1) {
    bigrams.push(`${words[index]}_${words[index + 1]}`);
  }

  return [...words, ...bigrams];
}

function buildVocabulary(examples) {
  const counts = new Map();

  for (const example of examples) {
    for (const token of example.tokens) {
      counts.set(token, (counts.get(token) ?? 0) + 1);
    }
  }

  return [...counts.entries()]
    .filter(([, count]) => count >= MIN_TOKEN_COUNT)
    .sort((left, right) => left[0].localeCompare(right[0]))
    .map(([token]) => token);
}

function trainNaiveBayes(examples, vocabulary, target, labels) {
  const vocabularySet = new Set(vocabulary);
  const classStats = {};

  for (const label of labels) {
    classStats[label] = {
      example_count: 0,
      token_total: 0,
      token_counts: {}
    };
  }

  for (const example of examples) {
    const stats = classStats[example[target]];
    stats.example_count += 1;

    for (const token of example.tokens) {
      if (!vocabularySet.has(token)) {
        continue;
      }

      stats.token_total += 1;
      stats.token_counts[token] = (stats.token_counts[token] ?? 0) + 1;
    }
  }

  const classes = {};

  for (const label of labels) {
    const stats = classStats[label];
    const denominator = stats.token_total + vocabulary.length;
    const tokenLogLikelihoods = {};

    for (const token of vocabulary) {
      tokenLogLikelihoods[token] = Math.log(((stats.token_counts[token] ?? 0) + 1) / denominator);
    }

    classes[label] = {
      log_prior: Math.log((stats.example_count + 1) / (examples.length + labels.length)),
      unknown_log_likelihood: Math.log(1 / denominator),
      token_log_likelihoods: tokenLogLikelihoods
    };
  }

  return { labels, classes };
}

function evaluateTrainingAccuracy(examples, classifiers) {
  const metrics = {};

  for (const target of TARGETS) {
    let correct = 0;

    for (const example of examples) {
      const prediction = predict(example.text, classifiers[target]);
      if (prediction.label === example[target]) {
        correct += 1;
      }
    }

    metrics[target] = {
      correct,
      total: examples.length,
      accuracy: Math.round((correct / examples.length) * 1000) / 1000
    };
  }

  return metrics;
}

function predict(text, classifier) {
  const tokens = tokenize(text);
  const scores = classifier.labels.map((label) => {
    const classModel = classifier.classes[label];
    let score = classModel.log_prior;

    for (const token of tokens) {
      score += classModel.token_log_likelihoods[token] ?? classModel.unknown_log_likelihood;
    }

    return { label, score };
  });

  scores.sort((left, right) => right.score - left.score);
  return scores[0];
}

function printSummary(model) {
  console.log(`Trained ${model.model_type} router model.`);
  console.log(`Examples: ${model.training_examples}`);
  console.log(`Vocabulary: ${model.vocabulary.length}`);

  for (const target of TARGETS) {
    const metric = model.metrics[target];
    console.log(`${target}: ${metric.correct}/${metric.total} training accuracy=${metric.accuracy}`);
  }

  console.log(`Wrote ${path.relative(PROJECT_ROOT, MODEL_PATH)}`);
}

main();
