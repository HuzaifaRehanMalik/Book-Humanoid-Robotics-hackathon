import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';

interface UserPreferences {
  adaptive_difficulty: 'beginner' | 'intermediate' | 'advanced';
  adaptive_code_samples: boolean;
  preferred_language: string;
  preferred_topics: string[];
}

const UserSettings: React.FC = () => {
  const { user, loading, updateUserPreferences } = useAuth();
  const [preferences, setPreferences] = useState<UserPreferences>({
    adaptive_difficulty: 'intermediate',
    adaptive_code_samples: true,
    preferred_language: 'en',
    preferred_topics: [],
  });
  const [saved, setSaved] = useState(false);

  // Load saved preferences from localStorage when user is available
  useEffect(() => {
    if (user) {
      const savedPrefs = localStorage.getItem('userPreferences');
      if (savedPrefs) {
        try {
          setPreferences(JSON.parse(savedPrefs));
        } catch (e) {
          console.error('Error parsing user preferences:', e);
        }
      }
    }
  }, [user]);

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement | HTMLInputElement>) => {
    const { name, value, type } = e.target;
    const checked = type === 'checkbox' ? (e.target as HTMLInputElement).checked : undefined;

    setPreferences(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));

    setSaved(false);
  };

  const handleTopicChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { value, checked } = e.target;
    setPreferences(prev => {
      const topics = prev.preferred_topics.includes(value)
        ? prev.preferred_topics.filter(topic => topic !== value)
        : [...prev.preferred_topics, value];

      return { ...prev, preferred_topics: topics };
    });

    setSaved(false);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      await updateUserPreferences(preferences);
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (error) {
      console.error('Error saving preferences:', error);
    }
  };

  if (loading) {
    return <div>Loading...</div>;
  }

  if (!user) {
    return <div>Please log in to access settings.</div>;
  }

  return (
    <div className="user-settings-container">
      <h2>User Settings & Preferences</h2>

      {saved && (
        <div className="alert alert--success">
          Preferences saved successfully!
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="adaptive_difficulty">Learning Difficulty:</label>
          <select
            id="adaptive_difficulty"
            name="adaptive_difficulty"
            value={preferences.adaptive_difficulty}
            onChange={handleChange}
            className="form-control"
          >
            <option value="beginner">Beginner</option>
            <option value="intermediate">Intermediate</option>
            <option value="advanced">Advanced</option>
          </select>
          <p className="help-text">
            This will adjust the complexity of explanations in the textbook content.
          </p>
        </div>

        <div className="form-group">
          <label className="checkbox-label">
            <input
              type="checkbox"
              name="adaptive_code_samples"
              checked={preferences.adaptive_code_samples}
              onChange={handleChange}
            />
            Include code examples in explanations
          </label>
        </div>

        <div className="form-group">
          <label htmlFor="preferred_language">Preferred Language:</label>
          <select
            id="preferred_language"
            name="preferred_language"
            value={preferences.preferred_language}
            onChange={handleChange}
            className="form-control"
          >
            <option value="en">English</option>
            <option value="ur">Urdu</option>
          </select>
        </div>

        <div className="form-group">
          <label>Preferred Topics:</label>
          <div className="checkbox-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                value="ros2"
                checked={preferences.preferred_topics.includes('ros2')}
                onChange={handleTopicChange}
              />
              ROS 2 Fundamentals
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                value="simulation"
                checked={preferences.preferred_topics.includes('simulation')}
                onChange={handleTopicChange}
              />
              Simulation (Gazebo/Unity)
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                value="ai_integration"
                checked={preferences.preferred_topics.includes('ai_integration')}
                onChange={handleTopicChange}
              />
              AI Integration
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                value="control_systems"
                checked={preferences.preferred_topics.includes('control_systems')}
                onChange={handleTopicChange}
              />
              Control Systems
            </label>
          </div>
        </div>

        <button type="submit" className="button button--primary">
          Save Preferences
        </button>
      </form>
    </div>
  );
};

export default UserSettings;