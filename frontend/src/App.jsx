import './App.css'
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import NavBar from "./components/NavBar";
import HeroSection from "./components/HeroSection";
import FeaturesSection from './components/FeaturesSection';
import UniqueClues from './components/UniqueClues';
import SetCarding from './components/SetCarding';
import Footer from "./components/Footer";
import FadeInSection from './components/FadeInSection';
import About from './components/About';
import QuestionGenerator from './components/QuestionGenerator';
import Register from './components/Register';
import Login from './components/Login';
import mesh from './assets/mesh.svg';

function App() {
  return (
    <Router>
      <div className="w-screen min-h-[1000px] fixed z-10 flex justify-center px-6 py-40 pointer-events-none">
        <img src={ mesh } className="opacity-60 absolute bottom-1 h-[900px]"/>
      </div>
      <div className="relative z-20">
        <NavBar />
        <div className="container mx-auto">
          <Routes>
            <Route
              path="/"
              element={
                <>
                  <FadeInSection>
                    <HeroSection />
                  </FadeInSection>
                  <FadeInSection>
                    <FeaturesSection />
                  </FadeInSection>
                  <FadeInSection>
                    <Footer />
                  </FadeInSection>
                </>
              }
            />
            <Route path="/unique-clues/" element={<UniqueClues />} />
            <Route path="/set-carding/" element={<SetCarding />} />
            <Route path="/question-generator/" element={<QuestionGenerator />} />
            <Route path="/about/" element={<About />} />
            <Route path="/login/" element={<Login />} />
            <Route path="/register/" element={<Register />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App
