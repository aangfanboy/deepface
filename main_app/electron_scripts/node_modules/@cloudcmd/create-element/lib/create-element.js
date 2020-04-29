'use strict';

const currify = require('currify');
const query = (a) => document.querySelector(`[data-name="${a}"]`);

const setAttribute = currify((el, obj, name) => el.setAttribute(name, obj[name]));
const set = currify((el, obj, name) => el[name] = obj[name]);
const not = currify((f, a) => !f(a));
const isCamelCase = (a) => a != a.toLowerCase();

module.exports = (name, options = {}) => {
    const {
        dataName,
        notAppend,
        parent = document.body,
        uniq = true,
        ...restOptions
    } = options;
    
    const elFound = isElementPresent(dataName);
    
    if (uniq && elFound)
        return elFound;
    
    const el = document.createElement(name);
    
    if (dataName)
        el.dataset.name = dataName;
    
    Object.keys(restOptions)
        .filter(isCamelCase)
        .map(set(el, options));
    
    Object.keys(restOptions)
        .filter(not(isCamelCase))
        .map(setAttribute(el, options));
    
    if (!notAppend)
        parent.appendChild(el);
    
    return el;
};

module.exports.isElementPresent = isElementPresent;

function isElementPresent(dataName) {
    if (!dataName)
        return;
    
    return query(dataName);
}

