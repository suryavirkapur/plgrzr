import { auth } from "@/lib/auth"; //import the auth client
import { useState } from "react";

export default function SignUp() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");

  const signUp = async () => {
    await auth.signUp.email(
      {
        email,
        password,
        name,
        image: undefined,
      },
      {
        onRequest: () => {
          <>Loading...</>;
        },
        onSuccess: () => {
          alert("Signed Up");
        },
        onError: (ctx) => {
          alert(ctx.error.message);
        },
      }
    );
  };

  return (
    <div>
      <label>Name</label>
      <input
        type="name"
        value={name}
        onChange={(e) => setName(e.target.value)}
      />{" "}
      <label>Password</label>
      <input
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />{" "}
      <label>Email</label>
      <input
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
      />
      <button onClick={signUp}>Sign Up</button>
    </div>
  );
}
