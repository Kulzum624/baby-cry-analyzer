import { useState, useEffect } from 'react';
import { 
  Container, 
  Box, 
  Typography, 
  Paper, 
  Button, 
  CircularProgress,
  Alert,
  ThemeProvider,
  createTheme,
  Tabs,
  Tab,
  IconButton,
  Card,
  CardContent,
  Grid,
  useMediaQuery,
  alpha
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import MicIcon from '@mui/icons-material/Mic';
import StopIcon from '@mui/icons-material/Stop';
import BabyChangingStationIcon from '@mui/icons-material/BabyChangingStation';
import { styled } from '@mui/material/styles';

// Mock function to simulate cry analysis
const analyzeCry = async (audioData: Blob | File): Promise<string> => {
  // Simulate API call delay
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  const categories = ['Hungry', 'Belly Pain', 'Burping', 'Discomfort', 'Tired'];
  return categories[Math.floor(Math.random() * categories.length)];
};

const VisuallyHiddenInput = styled('input')`
  clip: rect(0 0 0 0);
  clip-path: inset(50%);
  height: 1px;
  overflow: hidden;
  position: absolute;
  bottom: 0;
  left: 0;
  white-space: nowrap;
  width: 1px;
`;

const BabyCard = styled(Card)(({ theme }) => ({
  background: `linear-gradient(135deg, ${alpha(theme.palette.primary.light, 0.1)} 0%, ${alpha(theme.palette.secondary.light, 0.1)} 100%)`,
  border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
  transition: 'transform 0.2s ease-in-out',
  '&:hover': {
    transform: 'translateY(-4px)',
  },
}));

const theme = createTheme({
  palette: {
    primary: {
      main: '#FF69B4', // Soft pink
      light: '#FFB6C1',
      dark: '#DB7093',
    },
    secondary: {
      main: '#87CEEB', // Sky blue
      light: '#B0E0E6',
      dark: '#4682B4',
    },
    background: {
      default: '#FFF0F5', // Lavender blush
      paper: '#FFFFFF',
    },
  },
  typography: {
    fontFamily: '"Comic Sans MS", "Roboto", "Helvetica", "Arial", sans-serif',
    h3: {
      fontWeight: 600,
      color: '#FF69B4',
      textShadow: '2px 2px 4px rgba(0,0,0,0.1)',
    },
    h6: {
      color: '#FF69B4',
      fontWeight: 600,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 25,
          textTransform: 'none',
          fontWeight: 600,
          padding: '10px 24px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 6px 8px rgba(0,0,0,0.15)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 20,
          boxShadow: '0 8px 16px rgba(0,0,0,0.1)',
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          fontFamily: '"Comic Sans MS", "Roboto", "Helvetica", "Arial", sans-serif',
          fontWeight: 600,
          fontSize: '1.1rem',
        },
      },
    },
  },
});

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const cryTypes = [
  {
    title: 'Hungry',
    description: 'Short, low-pitched cries that rise and fall, often with lip smacking',
    tips: [
      'Check when your baby last ate',
      'Look for rooting or sucking motions',
      'Offer a feeding if it\'s been 2-3 hours'
    ]
  },
  {
    title: 'Belly Pain',
    description: 'Sharp, high-pitched cries with clenched fists and legs pulled up',
    tips: [
      'Try gentle tummy massage',
      'Do bicycle leg movements',
      'Check for gas or constipation'
    ]
  },
  {
    title: 'Burping',
    description: 'Short, rhythmic cries with pauses, often after feeding',
    tips: [
      'Hold baby upright against your shoulder',
      'Gently pat their back',
      'Try different burping positions'
    ]
  },
  {
    title: 'Discomfort',
    description: 'Fussy, intermittent cries with squirming movements',
    tips: [
      'Check diaper',
      'Adjust room temperature',
      'Look for tight clothing or tags'
    ]
  },
  {
    title: 'Tired',
    description: 'Whiny, nasal cries with eye rubbing and yawning',
    tips: [
      'Create a calm environment',
      'Try gentle rocking or swaying',
      'Use white noise if needed'
    ]
  }
];

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tabValue, setTabValue] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const isMobile = useMediaQuery('(max-width:600px)');

  useEffect(() => {
    return () => {
      if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
      }
    };
  }, [mediaRecorder, isRecording]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files[0]) {
      setFile(files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setAnalyzing(true);
    setError(null);
    try {
      const analysis = await analyzeCry(file);
      setResult(analysis);
    } catch (err) {
      setError('Failed to analyze the audio file. Please try again.');
    } finally {
      setAnalyzing(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.push(e.data);
        }
      };

      recorder.onstop = async () => {
        const audioBlob = new Blob(chunks, { type: 'audio/wav' });
        setAnalyzing(true);
        try {
          const analysis = await analyzeCry(audioBlob);
          setResult(analysis);
        } catch (err) {
          setError('Failed to analyze the audio. Please try again.');
        } finally {
          setAnalyzing(false);
        }
        stream.getTracks().forEach(track => track.stop());
      };

      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
      setResult(null);
      setError(null);
    } catch (err) {
      setError('Failed to access microphone. Please check your permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      setIsRecording(false);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
    setResult(null);
    setError(null);
  };

  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ 
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #FFF0F5 0%, #E6E6FA 100%)',
        py: 4,
        position: 'relative',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundImage: 'url("data:image/svg+xml,%3Csvg width=\'100\' height=\'100\' viewBox=\'0 0 100 100\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cpath d=\'M50 0C22.4 0 0 22.4 0 50s22.4 50 50 50 50-22.4 50-50S77.6 0 50 0zm0 90c-22.1 0-40-17.9-40-40s17.9-40 40-40 40 17.9 40 40-17.9 40-40 40z\' fill=\'%23FFB6C1\' fill-opacity=\'0.1\'/%3E%3C/svg%3E")',
          backgroundSize: '100px 100px',
          opacity: 0.5,
          zIndex: 0,
        }
      }}>
        <Container maxWidth="md" sx={{ position: 'relative', zIndex: 1 }}>
          <Box sx={{ 
            textAlign: 'center',
            mb: 4
          }}>
            <BabyChangingStationIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
            <Typography variant="h3" component="h1" gutterBottom>
              Baby Cry Analyzer
            </Typography>
            <Typography variant="subtitle1" color="text.secondary" sx={{ fontFamily: '"Comic Sans MS", cursive' }}>
              Understanding your little one's needs through cry analysis
            </Typography>
          </Box>
          
          <Paper elevation={3} sx={{ overflow: 'hidden', background: 'rgba(255, 255, 255, 0.9)' }}>
            <Tabs
              value={tabValue}
              onChange={handleTabChange}
              variant="fullWidth"
              sx={{ 
                borderBottom: 1, 
                borderColor: 'divider',
                '& .MuiTabs-indicator': {
                  backgroundColor: 'primary.main',
                  height: 3,
                }
              }}
            >
              <Tab label="Upload Audio" />
              <Tab label="Live Recording" />
            </Tabs>

            <TabPanel value={tabValue} index={0}>
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                <Button
                  component="label"
                  variant="contained"
                  startIcon={<CloudUploadIcon />}
                  disabled={analyzing}
                  size="large"
                  sx={{ 
                    minWidth: 200,
                    background: 'linear-gradient(45deg, #FF69B4 30%, #87CEEB 90%)',
                  }}
                >
                  Upload Audio File
                  <VisuallyHiddenInput
                    type="file"
                    accept="audio/*"
                    onChange={handleFileChange}
                  />
                </Button>

                {file && (
                  <Typography variant="body1" sx={{ mt: 2, fontFamily: '"Comic Sans MS", cursive' }}>
                    Selected file: {file.name}
                  </Typography>
                )}

                {file && !analyzing && !result && (
                  <Button
                    variant="contained"
                    color="secondary"
                    onClick={handleAnalyze}
                    fullWidth
                    size="large"
                    sx={{
                      background: 'linear-gradient(45deg, #87CEEB 30%, #FF69B4 90%)',
                    }}
                  >
                    Analyze Cry
                  </Button>
                )}
              </Box>
            </TabPanel>

            <TabPanel value={tabValue} index={1}>
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                <IconButton
                  onClick={isRecording ? stopRecording : startRecording}
                  sx={{
                    width: 100,
                    height: 100,
                    background: isRecording 
                      ? 'linear-gradient(45deg, #FF69B4 30%, #DB7093 90%)'
                      : 'linear-gradient(45deg, #87CEEB 30%, #4682B4 90%)',
                    '&:hover': {
                      background: isRecording 
                        ? 'linear-gradient(45deg, #DB7093 30%, #FF69B4 90%)'
                        : 'linear-gradient(45deg, #4682B4 30%, #87CEEB 90%)',
                    },
                    boxShadow: '0 4px 8px rgba(0,0,0,0.2)',
                  }}
                >
                  {isRecording ? <StopIcon sx={{ fontSize: 40, color: 'white' }} /> : <MicIcon sx={{ fontSize: 40, color: 'white' }} />}
                </IconButton>
                <Typography variant="body1" color="text.secondary" sx={{ fontFamily: '"Comic Sans MS", cursive' }}>
                  {isRecording ? 'Recording... Click to stop' : 'Click to start recording'}
                </Typography>
              </Box>
            </TabPanel>

            {(analyzing || result || error) && (
              <Box sx={{ p: 3, borderTop: 1, borderColor: 'divider' }}>
                {analyzing && (
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 2 }}>
                    <CircularProgress size={24} sx={{ color: 'primary.main' }} />
                    <Typography sx={{ fontFamily: '"Comic Sans MS", cursive' }}>Analyzing...</Typography>
                  </Box>
                )}

                {error && (
                  <Alert severity="error" sx={{ width: '100%', fontFamily: '"Comic Sans MS", cursive' }}>
                    {error}
                  </Alert>
                )}

                {result && (
                  <BabyCard sx={{ mt: 2 }}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
                        Analysis Result
                      </Typography>
                      <Typography variant="body1" sx={{ fontFamily: '"Comic Sans MS", cursive', mb: 2 }}>
                        Your baby is {result}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ fontFamily: '"Comic Sans MS", cursive' }}>
                        {cryTypes.find(type => type.title === result)?.description}
                      </Typography>
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="subtitle2" color="primary" gutterBottom>
                          What to do:
                        </Typography>
                        <Box component="ul" sx={{ pl: 2 }}>
                          {cryTypes.find(type => type.title === result)?.tips.map((tip, index) => (
                            <Typography component="li" key={index} variant="body2" sx={{ fontFamily: '"Comic Sans MS", cursive' }}>
                              {tip}
                            </Typography>
                          ))}
                        </Box>
                      </Box>
                    </CardContent>
                  </BabyCard>
                )}
              </Box>
            )}
          </Paper>

          <Grid container spacing={2} sx={{ mt: 4 }}>
            {cryTypes.map((type, index) => (
              <Grid item xs={12} sm={6} md={4} key={index}>
                <BabyCard>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      {type.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ fontFamily: '"Comic Sans MS", cursive' }}>
                      {type.description}
                    </Typography>
                  </CardContent>
                </BabyCard>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App; 