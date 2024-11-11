const functions = require('firebase-functions');
const { spawn } = require('child_process');

exports.app = functions.https.onRequest((req, res) => {
    const python = spawn('python', ['functions/app.py']);
    
    python.stdout.on('data', (data) => {
        res.send(data.toString());
    });
    
    python.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
        res.status(500).send(data.toString());
    });

    python.on('close', (code) => {
        console.log(`child process exited with code ${code}`);
    });
});
