import React from 'react';
import { useAuth } from '../../contexts/AuthContext';

const UserProfile: React.FC = () => {
  const { user, logout, loading } = useAuth();

  if (loading) {
    return <div>Loading...</div>;
  }

  if (!user) {
    return null; // Don't render anything if user is not logged in
  }

  return (
    <div className="user-profile">
      <div className="user-info">
        <h3>Welcome, {user.name}!</h3>
        <p className="user-email">{user.email}</p>
      </div>
      <button
        onClick={logout}
        className="button button--outline button--sm"
      >
        Logout
      </button>
    </div>
  );
};

export default UserProfile;