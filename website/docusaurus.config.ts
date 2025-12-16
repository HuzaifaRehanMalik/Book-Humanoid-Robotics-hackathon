import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'Physical AI & Humanoid Robotics Textbook',
  tagline: 'An AI-native textbook on Physical AI and Humanoid Robotics',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://physical-ai-humanoid-robotics.vercel.app',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'HuzaifaRehanMalik', // Usually your GitHub org/user name.
  projectName: 'Book-Humanoid-Robotics', // Usually your repo name.

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/Huzaifa-Rehan/Book-Humanoid-Robotics/tree/main/website/',
          // Add documentation-related settings
          routeBasePath: '/', // Serve docs from the root
          showLastUpdateTime: true,
          editCurrentVersion: true,
          // Enable next and previous navigation in the sidebar
        },
        theme: {
          customCss: './src/css/custom.css',
        },
        sitemap: {
          changefreq: 'weekly',
          priority: 0.5,
          ignorePatterns: ['/tags/**'],
          filename: 'sitemap.xml',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      logo: {
        alt: 'Physical AI & Humanoid Robotics Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'doc',
          docId: 'introduction',
          position: 'left',
          label: 'Textbook',
        },
        {
          href: 'https://github.com/HuzaifaRehanMalik/Book-Humanoid-Robotics',
          label: 'GitHub',
          position: 'right',
        },
        {
          href: 'https://panaversity.org',
          label: 'Panaverse',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Textbook',
          items: [
            {
              label: 'Introduction',
              to: '/introduction',
            },
            {
              label: 'ROS 2 Foundations',
              to: '/ros2-fundamentals',
            },
            {
              label: 'Gazebo & Unity Simulation',
              to: '/gazebo-unity-simulation',
            },
            {
              label: 'NVIDIA Isaac',
              to: '/nvidia-isaac',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'GitHub Repository',
              href: 'https://github.com/HuzaifaRehanMalik/Book-Humanoid-Robotics',
            },
            {
              label: 'Panaverse',
              href: 'https://panaversity.org',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Textbook Overview',
              to: '/introduction',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
    // Add search functionality
    algolia: {
      // The application ID provided by Algolia
      appId: process.env.ALGOLIA_APP_ID || 'YOUR_ALGOLIA_APP_ID',

      // Public API key: it is safe to commit it
      apiKey: process.env.ALGOLIA_API_KEY || 'YOUR_ALGOLIA_API_KEY',

      indexName: 'physical-ai-humanoid-robotics-textbook',

      contextualSearch: true,

      searchPagePath: 'search',
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
