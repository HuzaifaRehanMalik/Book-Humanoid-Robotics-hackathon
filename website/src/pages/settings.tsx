import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import UserSettings from '../components/Auth/UserSettings';
import UserProfile from '../components/Auth/UserProfile';
import LanguageToggle from '../components/LanguageToggle';
import { useAuth } from '../contexts/AuthContext';

function SettingsPage(): JSX.Element {
  const { user } = useAuth();

  return (
    <Layout
      title="Settings"
      description="User settings and preferences for the Physical AI & Humanoid Robotics textbook">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <div className="text--center margin-bottom--lg">
              <h1>User Settings & Preferences</h1>
              <p className="hero__subtitle">Customize your learning experience</p>
            </div>

            {user ? (
              <div className="card">
                <div className="card__header">
                  <h2>Account Settings</h2>
                </div>
                <div className="card__body">
                  <UserProfile />
                </div>
              </div>
            ) : (
              <div className="card">
                <div className="card__body">
                  <p>Please <Link to="/login">log in</Link> to access personalized settings.</p>
                </div>
              </div>
            )}

            <div className="margin-top--lg">
              <div className="card">
                <div className="card__header">
                  <h2>Personalization</h2>
                </div>
                <div className="card__body">
                  {user ? (
                    <UserSettings />
                  ) : (
                    <p>Log in to customize your learning experience.</p>
                  )}
                </div>
              </div>
            </div>

            <div className="margin-top--lg">
              <div className="card">
                <div className="card__header">
                  <h2>Language Preferences</h2>
                </div>
                <div className="card__body">
                  <LanguageToggle />
                  <p className="margin-top--sm">Select your preferred language for the interface.</p>
                </div>
              </div>
            </div>

            <div className="margin-top--lg text--center">
              <Link className="button button--secondary" to="/">
                Back to Textbook
              </Link>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default SettingsPage;