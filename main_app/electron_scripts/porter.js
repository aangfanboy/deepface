const net = require('net');
const struct = require('python-struct');
const prompt = require('electron-prompt');


function disableButtons(){
  var all=document.getElementsByTagName('input');
  var inp, i=0;
while(inp=all[i++]) {
inp.disabled=true;
}
}

function activateButtons(){
  var all=document.getElementsByTagName('input');
  var inp, i=0;
while(inp=all[i++]) {
inp.disabled=false;
}
}

function convertBlock(buffer){
  var buffer2 = struct.unpack("f", Buffer.from(buffer, "latin1"));
    return buffer2;
};

function mainWorker(){
  document.body.style.cursor = 'wait';
  // create a new tcp socket (not connected or listening yet)
  // socket is a stream and an event emitter
    client = net.connect({host:'127.0.0.1', port:64645},  () => {
            // 'connect' listener
            console.log('connected to server!');
            client.write('compare test.jpg test2.jpg');
          });

    client.on('data', function(data) {
    	console.log('Received: ' + convertBlock(data));
    	client.destroy(); // kill client after server's response
      document.body.style.cursor = 'auto';
    });

    client.on('close', function() {
    	console.log('Connection closed');
    });

  console.log("denendi");
}

function saveOutputs(){
  document.body.style.cursor = 'wait';
  disableButtons();
  path = document.getElementById("globalpath").text
  if (path != undefined){
  client = net.connect({host:'127.0.0.1', port:64645},  () => {
          // 'connect' listener
          console.log('connected to server!');
          const v1 = 'save_outputs ';
          client.write(v1.concat(path));
        });

    client.on('data', function(data) {
      path_saved = "saved_outputs/".concat(convertBlock(data)).concat(".json")
    	console.log('Saved To --> ' + path_saved);
      dialog.showMessageBox({
        title: "liyana feature saver",
          message: 'Saved',
          detail: 'Saved to \"'.concat(path_saved).concat("\""),
          buttons: ['Ok']
        })

    	client.destroy(); // kill client after server's response
      document.body.style.cursor = 'auto';
      activateButtons();
    });

    client.on('close', function() {
    	console.log('Connection closed');
    });

    }
}

function resetDB(){
  document.body.style.cursor = 'wait';
  disableButtons();
  client = net.connect({host:'127.0.0.1', port:64645},  () => {
          // 'connect' listener
          console.log('connected to server!');
          client.write('reset_database');
          dialog.showMessageBox({
            title: "liyana database manager",
              message: 'Done',
              detail: 'Database deleted and re-created',
              buttons: ['Ok']
            })
        });

    client.on('data', function(data) {
    	client.destroy(); // kill client after server's response
      document.body.style.cursor = 'auto';
      activateButtons();
    });

    client.on('close', function() {
    	console.log('Connection closed');
    });


}

function findWho(){
  document.body.style.cursor = 'wait';
  disableButtons();

  document.getElementById("d-name").textContent="Searching";
  document.getElementById("d-name").style.color = "#303F9F";
  document.getElementById("imagefromdb").src = "unknown.png";
  document.getElementById("dot3").style.backgroundColor  = "#c62828";

  delete json;
  load_json();

  path = document.getElementById("globalpath").text
  if (path != undefined){
  client = net.connect({host:'127.0.0.1', port:64645},  () => {
          // 'connect' listener
          console.log('connected to server!');
          const v1 = 'find_who ';
          client.write(v1.concat(path));
        });

    client.on('data', function(data) {
      const person_id = convertBlock(data)
      const whois = id2name(person_id);
        if(whois == "unknown"){
          document.getElementById("d-name").textContent="No Match - Unknown";
          document.getElementById("d-name").style.color = "#c62828";
          document.getElementById("imagefromdb").src = "unknown.png";
          document.getElementById("dot3").style.backgroundColor  = "#c62828";
        }
        else {
          document.getElementById("d-name").textContent="Match Found - " + whois;
          document.getElementById("d-name").style.color = "#00897B";
          document.getElementById("imagefromdb").src = id2photo(person_id);
          document.getElementById("dot3").style.backgroundColor  = "#00897B";
        }
      	console.log('This is: ' + whois);
      	client.destroy(); // kill client after server's response
        document.body.style.cursor = 'auto';
        activateButtons();
      }
    );

    client.on('close', function() {
    	console.log('Connection closed');
    });

    }
}


function savetoDatabase(){
  disableButtons();
  document.body.style.cursor = 'wait';
  path = document.getElementById("globalpath").text
  if (path != undefined){
    prompt({
    title: 'Prompt example',
    label: 'Name:',
    value: 'New Human',
    type: 'input'
})
.then((r) => {
    if(r === null) {
        console.log('user cancelled');
    } else {
       client = net.connect({host:'127.0.0.1', port:64645},  () => {
               // 'connect' listener
               console.log('connected to server!');
               const v1 = 'add_to_db ';
               client.write(v1.concat(path).concat(" ").concat(r));
               console.log('Saved As --> ' + r);
               dialog.showMessageBox({
                 title: "liyana database manager",
                   message: 'Saved',
                   detail: 'Person named '.concat(r).concat(' saved to database'),
                   buttons: ['Ok']
                 })

               delete json;
               load_json();
             });

         client.on('data', function(data) {
           console.log('Received: ' + convertBlock(data));
           client.destroy(); // kill client after server's response
           document.body.style.cursor = 'auto';
           activateButtons();
         });

         client.on('close', function() {
           console.log('Connection closed');
         });
    }
})
.catch(console.error);


    }
}


function display2dspace(){
disableButtons();
  document.body.style.cursor = 'wait';
  client = net.connect({host:'127.0.0.1', port:64645},  () => {
          // 'connect' listener
          console.log('connected to server!');

          client.write('create_2d_space');
        });

    client.on('data', function(data) {
      const liyana_path = "C:/Users/burak/Desktop"
      const face_path = liyana_path + "/liyana/main_app/python_server/2d_space.jpg"
      const my_info = convertBlock(data);
      if(my_info == -1 || my_info == "-1"){
        dialog.showErrorBox("Error", "There is an error occured when displaying 2D space. If you have less than 2 person in database, add more. If that doesn't work please check server's logs and open an issue.");
      }
      shell.openExternal(face_path)
      client.destroy(); // kill client after server's response
      document.body.style.cursor = 'auto';
      activateButtons();
    });

    client.on('close', function() {
      console.log('Connection closed');
    });


}


function goWebCam(path){
  disableButtons();
    document.body.style.cursor = 'wait';
    client = net.connect({host:'127.0.0.1', port:64645},  () => {
            // 'connect' listener
            console.log('connected to server!');
            dialog.showMessageBox({
              title: "liyana webcam manager",
                message: 'WebCam',
                detail: "Opening WebCam... Press ESC when you want to quit.",
                buttons: ['Ok']
              })

            console.log('go_for_webcam '.concat(path));
            client.write('go_for_webcam '.concat(path));
          });

      client.on('data', function(data) {
        const my_info = convertBlock(data);
        if(my_info == -1 || my_info == "-1"){
          dialog.showErrorBox("Error", "There is an error occured when opening WebCam. If that doesn't work please check server's logs and open an issue.");
        }
        client.destroy(); // kill client after server's response
        document.body.style.cursor = 'auto';
        activateButtons();
      });

      client.on('close', function() {
        console.log('Connection closed');
      });

}


function setfaceforPP(){
  disableButtons();
  document.body.style.cursor = 'wait';
  path = document.getElementById("globalpath").text
  if (path != undefined){
  client = net.connect({host:'127.0.0.1', port:64645},  () => {
          // 'connect' listener
          console.log('connected to server!');
          const v1 = 'give_face ';
          client.write(v1.concat(path));
        });

    client.on('data', function(data) {
      const liyana_path = "C:/Users/burak/Desktop"
      const face_path = liyana_path + "/liyana/main_app/python_server/faces2display/".concat(convertBlock(data)).concat(".jpg")
    	console.log('face saved in: ' + face_path);
      document.getElementById("my_image").src=face_path;
      document.getElementById("my_image").style.display = "show";
      document.getElementById("my_image").style.filter = "invert(0)";
    	client.destroy(); // kill client after server's response
    });

    client.on('close', function() {
    	console.log('Connection closed');
    });

    }
}


// mainWorker();
