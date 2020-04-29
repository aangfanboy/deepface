# Fullstore [![License][LicenseIMGURL]][LicenseURL] [![NPM version][NPMIMGURL]][NPMURL] [![Dependency Status][DependencyStatusIMGURL]][DependencyStatusURL] [![Build Status][BuildStatusIMGURL]][BuildStatusURL]

Functional variables.

## Install

```
npm i fullstore --save
```

## How to use?

```js
const fullstore = require('fullstore');
const user = fullstore();

const getValue = () => {
    return 'name';
};

user(getValue());

console.log(user());
// output
'name'
```

```js
const fullstore = require('fullstore');
const user = fullstore('hello');

console.log(user());
// output
'hello'
```

## Related

- [zames](https://github.com/coderaiser/zames "zames") - converts callback-based functions to Promises and apply currying to arguments

- [wraptile](https://github.com/coderaiser/wraptile "wraptile") - translate the evaluation of a function that takes multiple arguments into evaluating a sequence of 2 functions, each with a any count of arguments.

- [currify](https://github.com/coderaiser/currify "currify") - translate the evaluation of a function that takes multiple arguments into evaluating a sequence of functions, each with a single or more arguments.

## License

MIT

[NPMIMGURL]:                https://img.shields.io/npm/v/fullstore.svg?style=flat
[BuildStatusIMGURL]:        https://img.shields.io/travis/coderaiser/fullstore/master.svg?style=flat
[DependencyStatusIMGURL]:   https://img.shields.io/david/coderaiser/fullstore.svg?style=flat
[LicenseIMGURL]:            https://img.shields.io/badge/license-MIT-317BF9.svg?style=flat
[NPMURL]:                   https://npmjs.org/package/fullstore "npm"
[BuildStatusURL]:           https://travis-ci.org/coderaiser/fullstore  "Build Status"
[DependencyStatusURL]:      https://david-dm.org/coderaiser/fullstore "Dependency Status"
[LicenseURL]:               https://tldrlegal.com/license/mit-license "MIT License"

