import React from "react";
import "./Footer.css";
import { Button } from "../../other/Button";
import { Link } from "react-router-dom";
import {
  FaFacebook,
  FaYoutube,
  FaTwitter,
  FaLinkedin,
  FaHandMiddleFinger,
} from "react-icons/fa";

import { FiMail, FiDribbble } from "react-icons/fi";

function Footer() {
  return (
    <div className="footer-container">
      <div className="footer-links">
        <div className="footer-link-wrapper">
          <div className="footer-link-items">
            <h2 className="underline underline-offset-4">Account</h2>
            <Link to="/register">Register</Link>
            <Link to="/login">Login</Link>
            <Link to="/WIP">Profile</Link>
          </div>
          <div className="footer-link-items">
            <h2 className="underline underline-offset-4">Predictions</h2>
            <Link to="/predictions">Today's Games</Link>
            <Link to="/tomorrow">Tomorrow's Games</Link>
            <Link to="/future">Future Games</Link>
            <Link to="/history">History</Link>
          </div>
        </div>
        <div className="footer-link-wrapper">
          <div className="footer-link-items">
            <h2 className="underline underline-offset-4">Tools</h2>
            <Link to="/WIP">Rankings</Link>
            <Link to="/WIP">Comparison Tool</Link>
            <Link to="/calc/payout">Calculators</Link>
          </div>
          <div className="footer-link-items">
            <h2 className="underline underline-offset-4">General</h2>
            <Link to="/WIP">About</Link>
            <Link to="/WIP">FAQ</Link>
            <Link to="/WIP">Team Pages</Link>
            <Link></Link>
          </div>
        </div>
      </div>
      <section className="social-media">
        <div className="social-media-wrap">
          <div className="footer-logo">
            <Link to="/" className="social-logo">
              <FiDribbble className="navbar-icon" />
              RORACLE
            </Link>
          </div>
          <small className="website-rights">RORACLE © 2022</small>
          <div className="social-icons">
            <Link
              className="social-icon-link"
              to="/WIP"
              target="_blank"
              aria-label="Email"
            >
              <FiMail />
            </Link>
            <Link
              className="social-icon-link"
              to="/WIP"
              target="_blank"
              aria-label="Facebook"
            >
              <FaFacebook />
            </Link>
            <Link
              className="social-icon-link"
              to="/WIP"
              target="_blank"
              aria-label="Youtube"
            >
              <FaYoutube />
            </Link>
            <Link
              className="social-icon-link"
              to="/WIP"
              target="_blank"
              aria-label="Twitter"
            >
              <FaTwitter />
            </Link>
            <Link
              className="social-icon-link"
              to="/WIP"
              target="_blank"
              aria-label="LinkedIn"
            >
              <FaLinkedin />
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Footer;
