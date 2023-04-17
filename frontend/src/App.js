import React, { useState } from 'react';
import {
  Box,
  Button,
  Container,
  CssBaseline,
  TextField,
  Typography,
  Paper,
  Switch,
  FormGroup,
  FormControlLabel,
} from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';

function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [objective, setObjective] = useState('');
  const [chatHistory, setChatHistory] = useState([]);

  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
    },
  });

  const handleToggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  // Implement the run function to call your Flask API endpoints here
  const run = async () => {
    // Set the objective
    await fetch('http://127.0.0.1:5000/api/set_objective', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ objective }),
    });
  
    // Add the initial task
    await fetch('http://127.0.0.1:5000/api/add_initial_task', { method: 'POST' });
  
    while (true) {
      // Execute the next task
      const response = await fetch('http://127.0.0.1:5000/api/execute_next_task');
      const data = await response.json();
  
      if (!data.task || !data.result) {
        setChatHistory((prevChatHistory) => [
          ...prevChatHistory,
          '*****ALL TASKS COMPLETE*****',
        ]);
        break;
      }
  
      setChatHistory((prevChatHistory) => [
        ...prevChatHistory,
        '*****TASK LIST*****',
        `*****NEXT TASK*****\n${data.task.task_id}: ${data.task.task_name}`,
        `*****RESULT*****\n${data.result}`,
      ]);
  
      await new Promise((resolve) => setTimeout(resolve, 1000)); // Sleep for 1 second
    }
  };
  
  

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="sm">
        <Box sx={{ my: 4 }}>
          <Typography variant="h4" component="h1" gutterBottom>
            Babyagi Front-end
          </Typography>
          <FormGroup>
            <FormControlLabel
              control={<Switch checked={darkMode} onChange={handleToggleDarkMode} />}
              label="Toggle Dark Mode"
            />
          </FormGroup>
          <TextField
            fullWidth
            label="Enter Objective"
            value={objective}
            onChange={(e) => setObjective(e.target.value)}
          />
          <Box mt={2}>
            <Button variant="contained" color="primary" onClick={run}>
              Run Babyagi
            </Button>
          </Box>
          <Box mt={2}>
            <Paper elevation={3} style={{ padding: '16px', maxHeight: '300px', overflowY: 'auto' }}>
              {chatHistory.map((message, index) => (
                <Typography key={index} gutterBottom>
                  {message}
                </Typography>
              ))}
            </Paper>
          </Box>
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;
