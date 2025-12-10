import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import Signup from '../components/Auth/Signup';

function SignupPage(): JSX.Element {
  return (
    <Layout
      title="Sign Up"
      description="Create an account for the Physical AI & Humanoid Robotics textbook">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--6 col--offset-3">
            <div className="text--center margin-bottom--lg">
              <h1>Create an Account</h1>
              <p className="hero__subtitle">Get access to personalized learning features</p>
            </div>

            <div className="card">
              <div className="card__body">
                <Signup />
              </div>
              <div className="card__footer text--center">
                <p>
                  Already have an account? <Link to="/login">Log in</Link>
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default SignupPage;