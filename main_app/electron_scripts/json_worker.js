const { shell } = require('electron')

function nocache(module) {require("fs").watchFile(require("path").resolve(module), () => {delete require.cache[require.resolve(module)]})}
function age_dict(x){
  return (x*5).toString().concat("-").concat(((x*5)+5).toString());
}

function load_json(){
  nocache('../python_server/database.json');
    json = require('../python_server/database.json'); //(with path)
    sex_dict = {0: "man", 1: "woman"};
    eth_dict = {0: "white", 1: "black", 2: "asian", 3: "indian", 4: "Others"};

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

function id2sex(id){
  if(id == -1 || id == "-1"){
    return "unknown";
  }
  return sex_dict[json[id]["sex"]];
}

function id2age(id){
  if(id == -1 || id == "-1"){
    return "unknown";
  }
  return age_dict(parseInt(json[id]["age"]));
}

function id2eth(id){
  if(id == -1 || id == "-1"){
    return "unknown";
  }
  return eth_dict[json[id]["eth"]];
}


function openjsoninExp(){
  const liyana_path = "C:/Users/burak/Desktop"
  shell.openExternal(liyana_path+"/liyana/main_app/python_server/database.json")

}

load_json();
