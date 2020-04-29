const { app, BrowserWindow } = require("electron");
const url = require("url");

function newApp() {
  win = new BrowserWindow({
      webPreferences: {
          nodeIntegration: true
      },
      width: 1680,
       height: 980,
  });
  win.loadURL(
    url.format({
      pathname: "index.html",
      slashes: true
    })
  );

  // win.setMenu(null)
}

app.on("ready", newApp);
