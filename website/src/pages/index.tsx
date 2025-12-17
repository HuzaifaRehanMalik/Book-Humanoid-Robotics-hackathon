import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <div className="hero-inner">
          <Heading as="h1" className="hero__title">
            {siteConfig.title}
          </Heading>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link
              className="button button--primary button--lg"
              to="/introduction">
              Start Learning - 10min ⏱️
            </Link>
            <Link
              className="button button--secondary button--lg"
              to="/introduction">
              Textbook Overview
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

function ModuleCard({ title, description, to, icon }: { title: string; description: string; to: string; icon: string }) {
  return (
    <div className="card">
      <div className="card__header">
        <h3 className="card__title">{icon} {title}</h3>
      </div>
      <div className="card__body">
        <p>{description}</p>
        <Link to={to} className="button button--sm button--primary">
          Explore Module
        </Link>
      </div>
    </div>
  );
}

function ModulesSection() {
  const modules = [
    {
      title: 'ROS 2 Foundations',
      description: 'Learn the fundamentals of Robot Operating System 2, the standard framework for robotics development.',
      to: '/docs/ros2-fundamentals',
      icon: '🤖'
    },
    {
      title: 'Gazebo & Unity Simulation',
      description: 'Master simulation environments for testing and developing humanoid robotics applications.',
      to: '/docs/gazebo-unity-simulation',
      icon: '🎮'
    },
    {
      title: 'Nvidia Isaac & Omniverse',
      description: 'Explore advanced simulation and development tools for AI-powered robotics systems.',
      to: '/docs/nvidia-isaac',
      icon: '🌐'
    },
    {
      title: 'VLA & Conversational Robotics',
      description: 'Understand Vision-Language-Action models and how they enable human-robot interaction.',
      to: '/docs/vla-models',
      icon: '💬'
    },
    {
      title: 'Sensors, Motors & Kinematics',
      description: 'Study the physical components and mathematical models that enable robot movement.',
      to: '/docs/sensor-systems',
      icon: '⚙️'
    },
    {
      title: 'Motion Planning & Control',
      description: 'Learn algorithms for robot navigation, path planning, and precise movement control.',
      to: '/docs/control-systems',
      icon: '🎯'
    },
    {
      title: 'Reinforcement Learning for Humanoids',
      description: 'Discover how AI agents learn complex behaviors through interaction with the environment.',
      to: '/docs/learning-algorithms',
      icon: '🧠'
    },
    {
      title: 'Capstone Project',
      description: 'Apply your knowledge in a comprehensive project integrating all concepts learned.',
      to: '/docs/conclusion',
      icon: '🎓'
    }
  ];

  return (
    <section className="container margin-vert--lg">
      <div className="text--center padding-horiz--md">
        <Heading as="h2">Textbook Modules</Heading>
        <p className="padding-horiz--md">
          A comprehensive curriculum covering all aspects of Physical AI and Humanoid Robotics
        </p>
      </div>
      <div className="card-container">
        {modules.map((module, index) => (
          <ModuleCard
            key={index}
            title={module.title}
            description={module.description}
            to={module.to}
            icon={module.icon}
          />
        ))}
      </div>
    </section>
  );
}

function CTASection() {
  return (
    <section className="container margin-vert--lg">
      <div className="text--center padding-horiz--md">
        <Heading as="h2">Ready to Start Your Journey in Physical AI & Humanoid Robotics?</Heading>
        <p className="padding-horiz--md">
          Join thousands of students and professionals learning cutting-edge robotics technologies.
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg margin-right--md"
            to="/introduction">
            Begin Learning
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="/docs/humanoid-robotics-basics">
            Explore Content
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Home`}
      description="Physical AI & Humanoid Robotics Textbook - An AI-native textbook on Physical AI and Humanoid Robotics">
      <HomepageHeader />
      <main>
        <ModulesSection />
        <CTASection />
      </main>
    </Layout>
  );
}
