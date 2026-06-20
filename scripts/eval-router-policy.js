const fs = require("fs");
const vm = require("vm");

const source = fs
  .readFileSync("src/app.js", "utf8")
  .replace(/init\(\)\.catch[\s\S]*$/, "");
const stub = {
  textContent: "",
  dataset: {},
  addEventListener() {},
  querySelectorAll() { return []; },
  appendChild() {},
  classList: { toggle() {} },
  innerHTML: "",
  value: "",
  disabled: false,
  scrollTop: 0,
  scrollHeight: 0
};
const context = {
  console,
  document: {
    querySelector() { return { ...stub }; },
    querySelectorAll() { return []; },
    createElement() { return { ...stub }; }
  },
  navigator: {
    platform: "MacIntel",
    clipboard: { writeText: async () => {} }
  },
  window: { location: { origin: "http://localhost:5173" } },
  TextDecoder,
  Set,
  Map,
  Math,
  JSON
};

vm.createContext(context);
vm.runInContext(`${source}\nthis.__routerTest = { state, analyzePrompt, selectAutoRoute };`, context);

const policyEval = JSON.parse(fs.readFileSync("data/router-policy-eval.json", "utf8"));
const router = context.__routerTest;
const modelDatabase = JSON.parse(fs.readFileSync("data/model-capabilities.json", "utf8"));
router.state.models = modelDatabase.models.map((model) => ({
  ...model,
  routing_profile: modelDatabase.routing_profiles?.[model.id] ?? null
}));
router.state.routerModel = JSON.parse(fs.readFileSync("data/router-model.json", "utf8"));
router.state.installedLocalModels = new Set(policyEval.installed_local_models);

let failures = 0;

for (const testCase of policyEval.cases) {
  const signals = router.analyzePrompt(testCase.prompt);
  const decision = router.selectAutoRoute(testCase.prompt);
  const routeScope = decision.error ? "abstain" : decision.model.local ? "local" : "cloud";
  const matches = (
    signals.difficulty === testCase.difficulty
    && routeScope === testCase.route_scope
    && (!testCase.model_id || decision.model?.id === testCase.model_id)
  );

  if (!matches) failures += 1;

  console.log(
    `${matches ? "PASS" : "FAIL"} ${JSON.stringify(testCase.prompt)} `
    + `difficulty=${signals.difficulty} route=${routeScope} model=${decision.model?.id ?? "none"}`
  );
}

console.log(`\nPolicy evaluation: ${policyEval.cases.length - failures}/${policyEval.cases.length} passed.`);

if (failures > 0) {
  process.exitCode = 1;
}
