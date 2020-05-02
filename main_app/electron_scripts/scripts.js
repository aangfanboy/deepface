const { dialog } = require('electron').remote
const { clipboard } = require('electron')

function get_face(path){

  return path;
}

function makeNewTab(path){
  document.getElementById("d-name").textContent="Searching";
  document.getElementById("d-name").style.color = "#303F9F";
  document.getElementById("imagefromdb").src = "unknown.png";

  document.getElementById("dot1").style.backgroundColor  = "#c62828";
  document.getElementById("dot2").style.backgroundColor  = "#c62828";
  document.getElementById("dot3").style.backgroundColor  = "#c62828";
  document.getElementById("dot4").style.backgroundColor  = "#c62828";

  const face = get_face(path);
  document.getElementById("globalpath").text = path
  setfaceforPP();
  disableButtons();
  document.body.style.cursor = 'wait';

  document.getElementById("dot2").style.backgroundColor  = "#00897B";
  document.getElementById("dot1").style.backgroundColor  = "#00897B";

  setTimeout(function(){
     findWho();
  }, 2000);
}


function uploadImageFile() {

  const result = dialog.showOpenDialog({
    properties: ['openFile'],   filters: [{ name: 'Images', extensions: ['jpg', 'png', 'jpeg'] }]
  });

  result.then(result => {
    makeNewTab(result.filePaths[0])
  })


}

function comparer(path){
  path2 = document.getElementById("globalpath").text
  if (path2 != undefined){
    disableButtons();
    document.body.style.cursor = 'wait';
    // create a new tcp socket (not connected or listening yet)
    // socket is a stream and an event emitter
      client = net.connect({host:'127.0.0.1', port:64645},  () => {
              // 'connect' listener
              console.log('connected to server!');
              client.write('compare ' + path + ' ' + path2);
            });

      client.on('data', function(data) {
        const distance = convertBlock(data);
      	console.log('Distance between :' + distance);
        dialog.showMessageBox({
          title: "liyana basic comparer",
            message: 'Compared --> '.concat(distance.toString()),
            detail: 'Distance between '.concat(path).concat(' - ').concat(path2).concat('--> ').concat(distance.toString()),
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
  else {
    return 0.;
  }
}

function compareWith() {

  const { dialog } = require('electron').remote

  const result = dialog.showOpenDialog({
    properties: ['openFile'],   filters: [{ name: 'Images', extensions: ['jpg', 'png', 'jpeg'] }]
  });

  result.then(result => {
    comparer(result.filePaths[0])
  })


}


function goVideo() {

  const { dialog } = require('electron').remote

  const result = dialog.showOpenDialog({
    properties: ['openFile'],   filters: [{ name: 'Videos', extensions: ['gif', 'mp4'] }]
  });

  result.then(result => {
    goWebCam(result.filePaths[0])
  })


}


function donateButton(){
  clipboard.writeText("1LUFWnzrGVLdsZ7gnfee87iX6QqSn24Tvr")
  dialog.showMessageBox({
    title: "liyana - donate",
      message: 'Copied to Clipboard',
      detail: 'My Bitcoin/coin.space: 1LUFWnzrGVLdsZ7gnfee87iX6QqSn24Tvr\nThank you for donations, i am grateful.',
      buttons: ['Ok'],
    })
}
