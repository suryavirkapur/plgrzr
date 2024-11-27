import SignUp from "./components/SignUp";
import Login from "./components/Login";
import { auth } from "@/lib/auth";
import PdfUpload from "./components/Upload";
import PDFAnalysisDashboard from "./components/Compare";
import { ThemeProvider } from "./components/theme-provider";
import { ModeToggle } from "./components/mode-toggle";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./components/ui/tabs";
import { LogOut } from "lucide-react";
import { useState, useEffect } from "react";

import "./App.css";

const App = () => {
  const session = auth.useSession();
  const [mounted, setMounted] = useState(false);

  // Prevent hydration issues with theme
  useEffect(() => {
    setMounted(true);
  }, []);

  const handleLogout = async () => {
    try {
      await auth.signOut();
    } catch (error) {
      console.error("Logout failed:", error);
    }
  };

  if (!mounted) {
    return null;
  }

  return (
    <ThemeProvider defaultTheme="system" storageKey="ui-theme">
      <div className="min-h-screen bg-background">
        <header className="border-b">
          <div className="container mx-auto px-4 py-4 flex justify-between items-center">
            <h1 className="text-2xl font-bold">plgrzr</h1>
            <div className="flex items-center gap-4">
              <ModeToggle />
              {session.data && (
                <Button variant="ghost" size="icon" onClick={handleLogout}>
                  <LogOut className="h-5 w-5" />
                </Button>
              )}
            </div>
          </div>
        </header>

        <main className="container mx-auto px-4 py-8">
          {!session.data ? (
            <Card className="max-w-md mx-auto">
              <CardHeader>
                <CardTitle>Welcome to plgrzr</CardTitle>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="login">
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="login">Login</TabsTrigger>
                    <TabsTrigger value="signup">Sign Up</TabsTrigger>
                  </TabsList>
                  <TabsContent value="login">
                    <Login />
                  </TabsContent>
                  <TabsContent value="signup">
                    <SignUp />
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-8">
              <PDFAnalysisDashboard />

              <Card>
                <CardContent className="p-6">
                  <pre className="text-sm bg-muted p-4 rounded-lg overflow-auto">
                    {JSON.stringify(session, null, 2)}
                  </pre>
                </CardContent>
              </Card>

              {/* <PdfUpload /> */}
            </div>
          )}
        </main>
      </div>
    </ThemeProvider>
  );
};

export default App;
