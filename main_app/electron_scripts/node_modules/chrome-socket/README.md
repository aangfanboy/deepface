# chrome-socket

chrome tcp socket interface with streaming powers

## use

```javascript
var Socket = require('chrome-socket');

// create a new tcp socket (not connected or listening yet)
// socket is a stream and an event emitter
var socket = new Socket();

socket.connect(host, port);

// socket has connected
socket.once('connect', function() {
});

socket.on('data', function(chunk) {
    // chunck is an ArrayBuffer of data received from the socket
});

socket.on('error', function(err) {
    // uh oh!
});
```

## API

The API is modeled after node.js sockets. However, it emits and deals with `ArrayBuffer` objects versus node `Buffers` and provides no encoding. Encoding is provided by piping the socket to other streams.

### methods

#### connect

#### write

#### end

### events

#### connect

#### data

#### error

## install

Use [npm](https://npmjs.org) to install.

```
npm install chrome-socket
```
