/**
 * Import function triggers from their respective submodules:
 *
 * const {onCall} = require("firebase-functions/v2/https");
 * const {onDocumentWritten} = require("firebase-functions/v2/firestore");
 *
 * See a full list of supported triggers at https://firebase.google.com/docs/functions
 */

const {onRequest} = require("firebase-functions/v2/https");
const logger = require("firebase-functions/logger");

const functions = require("firebase-functions");
const admin = require("firebase-admin");
const express = require("express");
const app = express();

// Initialize Firebase Admin SDK
admin.initializeApp();

// Middleware to check if the user is authenticated
const checkAuth = async (req, res, next) => {
  const idToken = req.headers.authorization && req.headers.authorization.split('Bearer ')[1];
  
  if (!idToken) {
    return res.redirect("/login.html"); // Redirect to login if no token is provided
  }

  try {
    // Verify the token
    const decodedToken = await admin.auth().verifyIdToken(idToken);
    req.user = decodedToken; // Store user info in request object for future use
    next(); // Proceed to the requested page
  } catch (error) {
    console.error("Error verifying token:", error);
    return res.redirect("/login.html"); // Redirect to login if token verification fails
  }
};

// Apply the authentication middleware to all protected routes
app.use("/protected", checkAuth);

// Serve protected HTML pages from Firebase Hosting
app.get("/protected/*", (req, res) => {
  const page = req.params[0]; // Get the specific page requested
  res.sendFile(`/public/protected/${page}`, { root: __dirname });
});

// Export the app function
exports.app = functions.https.onRequest(app);


// Create and deploy your first functions
// https://firebase.google.com/docs/functions/get-started

// exports.helloWorld = onRequest((request, response) => {
//   logger.info("Hello logs!", {structuredData: true});
//   response.send("Hello from Firebase!");
// });
