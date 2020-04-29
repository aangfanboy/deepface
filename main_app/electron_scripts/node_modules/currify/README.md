# Currify [![License][LicenseIMGURL]][LicenseURL] [![NPM version][NPMIMGURL]][NPMURL] [![Dependency Status][DependencyStatusIMGURL]][DependencyStatusURL] [![Build Status][BuildStatusIMGURL]][BuildStatusURL]

Translate the evaluation of a function that takes multiple arguments into evaluating a sequence of functions, each with a single or more arguments.

## Install

```
npm i currify --save
```

## How to use?

```js
const currify = require('currify');

const mean = (a, b, c) => (a + b) / c;
const mean1 = currify(mean, 1);
const mean2 = mean1(2);

mean2(2);
// returns
1.5
```

## Related

- [fullstore](https://github.com/coderaiser/fullstore "fullstore") - functional variables.

- [zames](https://github.com/coderaiser/zames "zames") - converts callback-based functions to Promises and apply currying to arguments

- [wraptile](https://github.com/coderaiser/wraptile "wraptile") - translate the evaluation of a function that takes multiple arguments into evaluating a sequence of 2 functions, each with a any count of arguments.

## License

MIT

[NPMIMGURL]:                https://img.shields.io/npm/v/currify.svg?style=flat
[BuildStatusIMGURL]:        https://img.shields.io/travis/coderaiser/currify/master.svg?style=flat
[DependencyStatusIMGURL]:   https://img.shields.io/david/coderaiser/currify.svg?style=flat
[LicenseIMGURL]:            https://img.shields.io/badge/license-MIT-317BF9.svg?style=flat
[NPMURL]:                   https://npmjs.org/package/currify "npm"
[BuildStatusURL]:           https://travis-ci.org/coderaiser/currify  "Build Status"
[DependencyStatusURL]:      https://david-dm.org/coderaiser/currify "Dependency Status"
[LicenseURL]:               https://tldrlegal.com/license/mit-license "MIT License"
