const { shell } = require('electron')

function nocache(module) {require("fs").watchFile(require("path").resolve(module), () => {delete require.cache[require.resolve(module)]})}

function load_json(){
  nocache('../python_server/database.json');
    json = require('../python_server/database.json'); //(with path)

}

function id2name(id){
  if(id == -1 || id == "-1"){
    return "unknown";
  }

  console.log(json);
  return json[id]["name"];
}

function id2photo(id){

  if(id == -1 || id == "-1"){
    return "unknown";
  }
  return json[id]["face"];
}

function openjsoninExp(){
  const liyana_path = "C:/Users/burak/Desktop"
  shell.openExternal(liyana_path+"/liyana/main_app/python_server/database.json")

}

load_json();
