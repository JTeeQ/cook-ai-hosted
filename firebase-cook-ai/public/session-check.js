import { initializeApp } from "https://www.gstatic.com/firebasejs/10.13.2/firebase-app.js";
import { getAuth, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/10.13.2/firebase-auth.js";

      const firebaseConfig = {
        apiKey: "AIzaSyB1BRYzsdICZqEjCLei35VbGaJIs0Ns3x0",
        authDomain: "cook-ai-4fc88.firebaseapp.com",
        projectId: "cook-ai-4fc88",
        storageBucket: "cookai-acb31.appspot.com",
        messagingSenderId: "463225148454",
        appId: "1:21163147226:web:04ccf8a23cc82d78468ae3"
      };

      // Initialize Firebase
      const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// Function to check user session
function checkUserSession() {
    // Hide the page content by default
    document.body.style.display = 'none';

    // Listen for auth state changes
    onAuthStateChanged(auth, (user) => {
        if (user) {
            // User is signed in, allow access to the page
            document.body.style.display = 'block'; // Show page content
        } else {
            // User is not signed in, redirect to index.html
            window.location.href = 'login.html';
        }
    });
}

// Export the session checking function for use in other scripts
export { checkUserSession };