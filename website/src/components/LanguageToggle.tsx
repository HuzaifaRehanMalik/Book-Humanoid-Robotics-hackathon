import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import styles from './LanguageToggle.module.css';

const LanguageToggle: React.FC = () => {
  const [currentLanguage, setCurrentLanguage] = useState<string>('en');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const { user, updateUserPreferences } = useAuth();

  // Load saved language preference
  useEffect(() => {
    const savedLanguage = localStorage.getItem('preferredLanguage') || 'en';
    setCurrentLanguage(savedLanguage);
  }, []);

  const toggleLanguage = async (lang: string) => {
    // Update state
    setCurrentLanguage(lang);

    // Save to localStorage
    localStorage.setItem('preferredLanguage', lang);

    // If user is authenticated, update their preferences
    if (user && updateUserPreferences) {
      try {
        // Get existing preferences and update language
        const existingPrefs = localStorage.getItem('userPreferences');
        let userPrefs = existingPrefs ? JSON.parse(existingPrefs) : {};

        userPrefs = {
          ...userPrefs,
          preferred_language: lang
        };

        await updateUserPreferences(userPrefs);
      } catch (error) {
        console.error('Error updating user preferences:', error);
      }
    }

    // This would trigger a re-render with translated content in a real app
    console.log(`Language changed to: ${lang}`);

    setIsDropdownOpen(false);
  };

  const getLanguageName = (code: string): string => {
    const languages: Record<string, string> = {
      'en': 'English',
      'ur': 'Urdu'
    };
    return languages[code] || code;
  };

  return (
    <div className={styles.languageToggle}>
      <button
        className={styles.languageToggleButton}
        onClick={() => setIsDropdownOpen(!isDropdownOpen)}
        aria-label="Change language"
      >
        {getLanguageName(currentLanguage)}
      </button>

      {isDropdownOpen && (
        <div className={styles.languageToggleDropdown}>
          <button
            className={`${styles.languageOption} ${currentLanguage === 'en' ? styles.active : ''}`}
            onClick={() => toggleLanguage('en')}
          >
            English
          </button>
          <button
            className={`${styles.languageOption} ${currentLanguage === 'ur' ? styles.active : ''}`}
            onClick={() => toggleLanguage('ur')}
          >
            اردو
          </button>
        </div>
      )}
    </div>
  );
};

export default LanguageToggle;