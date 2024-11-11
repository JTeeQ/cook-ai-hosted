// Import Firebase modules
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.13.2/firebase-app.js";
import { getAuth, onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/10.13.2/firebase-auth.js";

// Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyB1BRYzsdICZqEjCLei35VbGaJIs0Ns3x0",
    authDomain: "cook-ai-4fc88.firebaseapp.com",  
    projectId: "cook-ai-4fc88",
    storageBucket: "cookai-acb31.appspot.com",
    messagingSenderId: "463225148454",
    appId: "1:21163147226:web:04ccf8a23cc82d78468ae3"
};

// Initialize Firebase and Authentication
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

function displayUserInfo(user) {
  const userInfoContainer = document.getElementById("user-info-container");

  if (userInfoContainer) {
    // Clear any previous content
    userInfoContainer.innerHTML = '';

    // Display the user's email
    const emailText = document.createElement('span');
    emailText.id = "user-email";
    emailText.style.fontSize = '20px';
    emailText.textContent = user.email;

    // Adjust color based on page path
    const path = window.location.pathname;
    if (path.endsWith('/index.html') || path === '/' || path.endsWith('/protected/kitchen.html')) {
      emailText.style.color = '#FFFFFF'; // White on index and kitchen
    } else if (path.endsWith('/protected/fridge.html')) {
      emailText.style.color = '#000000'; // Black on fridge
    } else {
      emailText.style.color = '#FF0000'; // Red for unrecognized path
    }

    // Create the sign-out button
    const signOutButton = document.createElement('button');
    signOutButton.innerText = 'Sign Out';
    signOutButton.style.cursor = 'pointer';
    signOutButton.onclick = () => handleSignOut();

    // Append email and button to the container
    userInfoContainer.appendChild(emailText);
    userInfoContainer.appendChild(signOutButton);
  }
}

// Handle Sign-Out
function handleSignOut() {
  signOut(auth)
    .then(() => {
      console.log("User signed out successfully");
      window.location.href = "/login.html";
    })
    .catch((error) => {
      console.error("Error signing out:", error);
    });
}

// Monitor Authentication State
onAuthStateChanged(auth, (user) => {
  const isLoginPage = window.location.pathname.endsWith('/login.html');

  if (user) {
    if (!isLoginPage) {
      displayUserInfo(user);
    }
  } else {
    const isProtectedPage = window.location.pathname.startsWith('/protected');
    if (isProtectedPage) {
      window.location.href = "/login.html";
    }
  }
});
