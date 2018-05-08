//=====================Global Vars==================================
var webSocket = null;
var image = null;
var acusArray;

function applicationViewModel() {

    var address = "ws://127.0.0.1:8092/ws";

    webSocket = new WebSocket(address);
    console.log("websocket started")

    /*     webSocket.onopen = function () {
            if (window.location.pathname == "/index.html" ||
                window.location.pathname == "/") {
                webSocket.send('register');
            } else if (window.location.pathname == "/detect.html") {
                webSocket.send('detect');
            }
        } */

    webSocket.onclose = function (e) {
        // webSocket = null;
        webSocket.send('disconnect');
    }

    webSocket.onmessage = function (e) {
        if (e.data.startsWith("cvoMessage")) {
            console.log('nothing')

        } else if (e.data.startsWith("registration")) {
            console.log("message from registration service. ");
			(function (){
				window.location.href = '/reg_complete';
			})();
        } else if (e.data != "Dropped Frame") {
            try {
                draw(e.data);
            }
            catch (error) { }

        } else if (e.data.startsWith("authReply")) {
            temp = e.data.split("|");
            msg = temp[1];
            if (msg == "Success") {
                document.getElementById("typeWriter").innerHTML = "";
                var welcomeElement = document.createElement("p");
                var welcomeMessage = document.createElement("h2");
                welcomeMessage.innerText = "Welcome, " + temp[2];
                welcomeMessage.id = "welcomeMessage";
                welcomeElement.appendChild(welcomeMessage);
                document.getElementById("typeWriter").appendChild(welcomeElement);
                alert("auth success")

                type();

            } else if (msg == "Failed") {
                alert("Authentication failed. Incorrect username/password combo.")
            } else {
                alert("Authentication failed. Unkown reason. ")
            }
        } else {
            draw(e.data);
        }
    }

}

function sendCompesInfo() {
    var id = document.getElementById('POST');
    var masterString = "sendCompes";
    for (var i = 0; i < id.elements.length; i++) {
        masterString += "|" + id.elements[i].value;
    }
    webSocket.send(masterString);
}

function sendAssignInfo() {
    var msg = "assign|";
    msg += $("#users option:selected").text() + "|";
    msg += $("#fingers option:selected").text() + "|";
    msg += $("#states option:selected").text();
    webSocket.send(msg);
}

function draw(input) {
    let image = document.getElementsByClassName("frame-display")[0];
    image.src = "data:image/jpg;base64," + input;
}


$(document).ready(function () {
    /*
    Description: This function specifies the behaviour of the program when the user starts the application.
    Inputs: an event related to the application opening
    Outputs: N\A
    Notes: This program sets up the knockout bindings and starts the python subprocess
           that houses the Twisted Client.
    */
    //Apply knockout data-bindings
    new applicationViewModel();
});
