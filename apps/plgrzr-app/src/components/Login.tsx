import { auth } from "@/lib/auth";
import { useState } from "react";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const login = async () => {
    const { data } = await auth.signIn.email(
      {
        email,
        password,
      },
      {
        onRequest: () => {
          <>Loading...</>;
        },
        onSuccess: () => {
          alert("Logged In");
        },
        onError: (ctx) => {
          alert(ctx.error.message);
        },
      }
    );
    const token = data?.session.id;
    // Store the token securely (e.g., in localStorage)
    localStorage.setItem("bearer_token", token || "");
  };

  return (
    <div>
      <label>Email</label>
      <input
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
      />

      <label>Password</label>
      <input
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />

      <button onClick={login}>Login</button>
    </div>
  );
}
