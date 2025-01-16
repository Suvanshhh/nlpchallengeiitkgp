// import { initializeApp } from "firebase/app";
// import { getAuth,setPersistence, browserLocalPersistence,} from "firebase/auth";
// import { getFirestore } from "firebase/firestore";

// // Import the functions you need from the SDKs you need
// // TODO: Add SDKs for Firebase products that you want to use
// // https://firebase.google.com/docs/web/setup#available-libraries

// // Your web app's Firebase configuration
// // For Firebase JS SDK v7.20.0 and later, measurementId is optional
// const firebaseConfig = {
//   apiKey: "AIzaSyDEiM791lSj8n-zhyMmQI27PWJscB6AzNE",
//   authDomain: "ai-powered-assistant-4e4f2.firebaseapp.com",
//   projectId: "ai-powered-assistant-4e4f2",
//   storageBucket: "ai-powered-assistant-4e4f2.firebasestorage.app",
//   messagingSenderId: "713469332465",
//   appId: "1:713469332465:web:aed43d0978f0945f5cd250",
//   measurementId: "G-G5QTVRQMF8"
// };

// // Initialize Firebase
// const app = initializeApp(firebaseConfig);
// export const auth = () => getAuth(app);
// export const db = () => getFirestore(app);

// setPersistence(auth, browserLocalPersistence).catch((err) =>{
//     console.log("Auth persistence erro", err);
// });

import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import {
  getAuth,
  setPersistence,
  browserLocalPersistence,
} from "firebase/auth";
import { getMessaging } from "firebase/messaging";

const firebaseConfig = {
    apiKey: "AIzaSyDEiM791lSj8n-zhyMmQI27PWJscB6AzNE",
    authDomain: "ai-powered-assistant-4e4f2.firebaseapp.com",
    projectId: "ai-powered-assistant-4e4f2",
    storageBucket: "ai-powered-assistant-4e4f2.firebasestorage.app",
    messagingSenderId: "713469332465",
    appId: "1:713469332465:web:aed43d0978f0945f5cd250",
    measurementId: "G-G5QTVRQMF8"
  };
// const firebaseConfig = {
//   apiKey: "AIzaSyAoOWMl-mopoWc_GLYseGamZpm_EtrBDDM",
//   authDomain: "master-class-97666.firebaseapp.com",
//   projectId: "master-class-97666",
//   storageBucket: "master-class-97666.appspot.com",
//   messagingSenderId: "507449038806",
//   appId: "1:507449038806:web:93d464aef6233d14f9b677",
// };

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);
export const messaging = getMessaging(app);

// Set persistence correctly using imported functions
setPersistence(auth, browserLocalPersistence).catch((error) => {
  console.error("Auth persistence error:", error);
});