# create-element [![License][LicenseIMGURL]][LicenseURL] [![NPM version][NPMIMGURL]][NPMURL] [![Dependency Status][DependencyStatusIMGURL]][DependencyStatusURL] [![Build Status][BuildStatusIMGURL]][BuildStatusURL] [![Coverage][CoverageIMGURL]][CoverageURL]

Create DOM element.

## Install

```
npm i @cloudcmd/create-element
```

## API

```js
const createElement = require('@cloudcmd/create-element');

// create DOM-element with no attributes
const div = createElement('div');

// add innerHTML
const el = createElement('div', {
    notAppend: false, // default
    parent: document.body, // default
    uniq: true, // default
    innerHTML: '<span></span>',
    className: 'abc',
});

// load css
const el = createElement('link', {
    rel: 'stylesheet',
    href: '/style.css',
    parent: document.head,
});
```

# License
MIT

[NPMIMGURL]:                https://img.shields.io/npm/v/@cloudcmd/create-element.svg?style=flat&longCache=true
[BuildStatusIMGURL]:        https://img.shields.io/travis/cloudcmd/create-element/master.svg?style=flat&longCache=true
[DependencyStatusIMGURL]:   https://img.shields.io/david/cloudcmd/create-element.svg?style=flat&longCache=true
[LicenseIMGURL]:            https://img.shields.io/badge/license-MIT-317BF9.svg?style=flat&longCache=true

[NPMURL]:                   https://npmjs.org/package/@cloudcmd/create-element "npm"
[BuildStatusURL]:           https://travis-ci.org/cloudcmd/create-element  "Build Status"
[DependencyStatusURL]:      https://david-dm.org/cloudcmd/create-element "Dependency Status"
[LicenseURL]:               https://tldrlegal.com/license/mit-license "MIT License"

[CoverageURL]:              https://coveralls.io/github/cloudcmd/create-element?branch=master
[CoverageIMGURL]:           https://coveralls.io/repos/cloudcmd/create-element/badge.svg?branch=master&service=github

