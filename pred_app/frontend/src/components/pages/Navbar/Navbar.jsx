import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { FaBars, FaTimes } from "react-icons/fa";
import { FiDribbble } from "react-icons/fi";
import { Button } from "../../other/Button";
import "./Navbar.css";
import { IconContext } from "react-icons/lib";

function Navbar(props) {
  const logout = async () => {
    await fetch("http://127.0.0.1:8000/api/logout/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
    });

    props.setUser("");
    console.log(props.user);
  };

  let menu;
  const [click, setClick] = useState(false);
  const [button, setButton] = useState(true);

  const handleClick = () => setClick(!click);
  const hideMenu = () => setClick(false);

  const showButton = () => {
    if (window.innerWidth <= 960) {
      setButton(false);
    } else {
      setButton(true);
    }
  };

  useEffect(() => {
    showButton();
    window.addEventListener("resize", showButton);
    return () => {
      window.removeEventListener("resize", showButton);
    };
  }, []);

  return (
    <IconContext.Provider value={{ color: "#fff" }}>
      <div className="navbar">
        <div className="navbar-container">
          <Link to="/" className="navbar-logo" onClick={hideMenu}>
            <FiDribbble className="navbar-icon" />
            ROBOOKIE
          </Link>
          <div className="menu-icon" onClick={handleClick}>
            {click ? <FaTimes /> : <FaBars />}
          </div>
          <div className="active">
            <ul className={click ? "nav-menu active" : "nav-menu"}>
              <li className="nav-item">
                <Link
                  to="/predictions"
                  className="nav-links"
                  onClick={hideMenu}
                >
                  Predictions
                </Link>
              </li>
              <li className="nav-item">
                <Link to="/future" className="nav-links" onClick={hideMenu}>
                  Future
                </Link>
              </li>
              <li className="nav-item">
                <Link to="/history" className="nav-links" onClick={hideMenu}>
                  History
                </Link>
              </li>
              <li className="nav-item">
                <Link to="/WIP" className="nav-links" onClick={hideMenu}>
                  Compare
                </Link>
              </li>
              <li className="nav-item">
                <Link
                  to="/calc/payout"
                  className="nav-links"
                  onClick={hideMenu}
                >
                  Calculators
                </Link>
              </li>
              <li className="nav-item">
                <Link to="/WIP" className="nav-links" onClick={hideMenu}>
                  FAQ
                </Link>
              </li>
              {props.user === "" ? (
                <li className="nav-btn">
                  {button ? (
                    <Link to="/login" className="btn-link">
                      <Button buttonStyle="btn--outline">Login</Button>
                    </Link>
                  ) : (
                    <Link to="/login" className="btn-link" onClick={hideMenu}>
                      <Button
                        buttonStyle="btn--outline"
                        buttonSize="btn--mobile"
                        className="navbar-btn"
                      >
                        Login
                      </Button>
                    </Link>
                  )}
                </li>
              ) : (
                <li className="nav-btn">
                  {button ? (
                    <Link to="#" className="btn-link" onClick={logout}>
                      <Button buttonStyle="btn--outline">Logout</Button>
                    </Link>
                  ) : (
                    <Link
                      to="#"
                      className="btn-link"
                      onClick={() => {
                        logout();
                        hideMenu();
                      }}
                    >
                      <Button
                        buttonStyle="btn--outline"
                        buttonSize="btn--mobile"
                        className="navbar-btn"
                      >
                        Logout
                      </Button>
                    </Link>
                  )}
                </li>
              )}
              {props.user === "" ? (
                <li className="nav-btn">
                  {button ? (
                    <Link to="/register" className="btn-link">
                      <Button
                        buttonStyle="btn--outline"
                        style={{
                          padding: "0px 0px",
                        }}
                      >
                        Register
                      </Button>
                    </Link>
                  ) : (
                    <Link
                      to="/register"
                      className="btn-link"
                      onClick={hideMenu}
                    >
                      <Button
                        buttonStyle="btn--outline"
                        buttonSize="btn--mobile"
                        className="navbar-btn btn--next"
                      >
                        Register
                      </Button>
                    </Link>
                  )}
                </li>
              ) : null}
            </ul>
          </div>
        </div>
      </div>
    </IconContext.Provider>
  );
}

export default Navbar;
