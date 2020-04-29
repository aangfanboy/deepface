var Stream = require('stream');
var inherits = require('inherits');

var Socket = function(options) {
    var self = this;
    Stream.call(self);

    self.writable = true;
    self.readable = true;

    self._socket_info = undefined;

    chrome.socket.create("tcp", {}, function(create_info) {
        self._socket_info = create_info;
        self.emit('_created');
    });

    // once connected, start trying to read
    self.once('connect', read.bind(self));
}
inherits(Socket, Stream);

/// write {ArrayBuffer} data to the socket
Socket.prototype.write = function(data) {
    var self = this;
    chrome.socket.write(self._socket_info.socketId, data, write_complete.bind(self));
};

/// write {ArrayBuffer} data and then end connection
Socket.prototype.end = function(data) {
    var self = this;

    if (data) {
        self.write(data);
    }

    chrome.socket.disconnect(self._socket_info.socketId);

    self.emit('end');
    self.emit('close');

    return self;
};

/// connect to the given port and host
Socket.prototype.connect = function(port, host) {
    var self = this;

    function connect() {
        var socket_id = self._socket_info.socketId;
        chrome.socket.connect(socket_id, host, port, function(error_code) {
            if (error_code < 0) {
                return self.emit('error', new Error('unable to connect: ' + error_code));
            }

            // connected
            self.emit('connect');
        });
    }

    // not yet created
    if (!self._socket_info) {
        return self.once('_created', connect);
    }

    connect();
};

// read from a socket
// 'this' should be a socket when called
function read() {
    var self = this;

    chrome.socket.read(self._socket_info.socketId, null, function(read_info) {
        if (read_info.resultCode < 0) {
            self.emit('error', new Error('tcp read failed: ' + read_info.resultCode));
            return;
        }

        self.emit('data', read_info.data);

        // read again
        read.bind(self)();
    });
};

function write_complete(write_info) {
    var self = this;
    if (write_info.bytesWritten <= 0) {
        self.emit('error', new Error('tcp socket write error'));
    }
};

/// events
//connect
//data
//end
//timeout
//drain
//error
//close

// set encoding?

module.exports = Socket;

