import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import Login from '../components/Auth/Login';

function LoginPage(): JSX.Element {
  return (
    <Layout
      title="Login"
      description="Login to your Physical AI & Humanoid Robotics textbook account">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--6 col--offset-3">
            <div className="text--center margin-bottom--lg">
              <h1>Login to Your Account</h1>
              <p className="hero__subtitle">Access personalized learning features</p>
            </div>

            <div className="card">
              <div className="card__body">
                <Login />
              </div>
              <div className="card__footer text--center">
                <p>
                  Don't have an account? <Link to="/signup">Sign up</Link>
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default LoginPage;