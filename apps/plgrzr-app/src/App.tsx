import SignUp from "./components/SignUp";
import Login from "./components/Login";
import { auth } from "./lib/auth";
import "./App.css";
import PdfUpload from "./components/Upload";

const App = () => {
  const session = auth.useSession();

  const handleLogout = async () => {
    try {
      await auth.signOut();
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  return (
    <div>
      <h1>plgrzr</h1>
      {JSON.stringify(session, null, 2)}
      <br />
      {!session.data ? (
        <>
          <SignUp />
          <Login />
        </>
      ) : (
        <>
          <PdfUpload />
          <button onClick={handleLogout}>Logout</button>
        </>
      )}
    </div>
  );
};

export default App;
