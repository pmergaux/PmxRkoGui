import express from 'express';
import http from 'http';
import { Server } from 'socket.io';
import path from 'path';

const app = express();
const server = http.createServer(app);
const io = new Server(server);

app.use(express.json());

// On utilise process.cwd() pour éviter les problèmes de __dirname
app.get('/', (req, res) => {
    res.sendFile(path.join(process.cwd(), 'index.html'));
});

app.post('/data', (req, res) => {
    console.log("Donnée reçue :", req.body);
    io.emit('nouveau_calcul', req.body);
    res.status(200).send("OK");
});

server.listen(5000, () => console.log("Serveur actif sur http://localhost:5000"));