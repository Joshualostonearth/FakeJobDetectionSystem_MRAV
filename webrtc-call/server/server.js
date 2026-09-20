const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');

const app = express();
// Serve static files from the public directory
app.use(express.static(path.join(__dirname, '../public')));

const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: '*',
  }
});

const PORT = 3000;
const ROOM = 'voice-room';

io.on('connection', (socket) => {
  console.log(`[Socket] User connected: ${socket.id}`);

  // When a user joins the room
  socket.on('join', () => {
    socket.join(ROOM);
    console.log(`[Socket] User ${socket.id} joined room: ${ROOM}`);
    // Notify others in the room that a new user joined
    socket.to(ROOM).emit('user-joined', socket.id);
  });

  // Relay WebRTC Offer
  socket.on('offer', (data) => {
    console.log(`[Signaling] Relaying offer from ${socket.id}`);
    socket.to(ROOM).emit('offer', data);
  });

  // Relay WebRTC Answer
  socket.on('answer', (data) => {
    console.log(`[Signaling] Relaying answer from ${socket.id}`);
    socket.to(ROOM).emit('answer', data);
  });

  // Relay ICE candidates
  socket.on('ice-candidate', (data) => {
    console.log(`[Signaling] Relaying ICE candidate from ${socket.id}`);
    socket.to(ROOM).emit('ice-candidate', data);
  });

  // Relay Hangup
  socket.on('hangup', () => {
    console.log(`[Signaling] User hung up: ${socket.id}`);
    socket.to(ROOM).emit('hangup', socket.id);
  });

  // Handle Disconnect
  socket.on('disconnect', () => {
    console.log(`[Socket] User disconnected: ${socket.id}`);
    socket.to(ROOM).emit('hangup', socket.id);
  });
});

// Listen on 0.0.0.0 to allow access from local network (phones, etc.)
server.listen(PORT, '0.0.0.0', () => {
  console.log(`[Server] Signaling server running on http://0.0.0.0:${PORT}`);
});
