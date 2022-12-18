import React, { useState } from "react";
import { AiOutlineArrowUp, AiOutlineArrowDown } from "react-icons/ai";
import {
  BsArrowUpCircle,
  BsArrowDownCircle,
  BsFillSignpost2Fill,
} from "react-icons/bs";
import "./GameCard.css";

const GameCard = (props) => {
  const [shownLineups, setShownLineups] = useState({});

  const toggleLineup = (index) => {
    setShownLineups((prevShownLineup) => ({
      ...prevShownLineup,
      [index]: !prevShownLineup[index],
    }));
  };

  const [show, setShow] = useState(true);
  const [games, setGames] = useState([]);
  const currentGames = props.games;

  if (!currentGames) {
    console.log("Waiting");
  }

  if (currentGames) {
    console.log(currentGames);
  }

  return (
    <>
      {currentGames.map((game) => (
        <div key={game.index} className="card">
          <div className="bottom-section">
            <div className="label">
              <h2 className="subheading border-b border-opacity-01 ">
                {game.Time + " ET"}
              </h2>
              <h2 className="heading">{game.A_Team}</h2>
              <p className="stat_name">{game.A_Record}</p>
              <p className="stat_name">{game.A_ELO}</p>
              <p className="stat_name">{game.A_NET}</p>
              <p className="stat_name highlight">{game.A_Pred + "%"}</p>
              <p className="stat_name">{game.A_Pred_ML}</p>
            </div>
            <div className="path">
              <h2 className="subheading border-b border-opacity-01 ">{`Game: ${
                game.index + 1
              }`}</h2>
              <div className="line"></div>
              <p className="stat_name">Record</p>
              <p className="stat_name">Elo Rating</p>
              <p className="stat_name">Net Rating</p>
              <p className="stat_name highlight">Prediction</p>
              <p className="stat_name">Moneyline</p>
            </div>
            <div className="label">
              <h2 className="subheading border-b border-opacity-01">
                {"@ " + game.Location}
              </h2>
              <h2 className="heading">{game.H_Team}</h2>
              <p className="stat_name">{game.H_Record}</p>
              <p className="stat_name">{game.H_ELO}</p>
              <p className="stat_name">{game.H_NET}</p>
              <p className="stat_name highlight">{game.H_Pred + "%"}</p>
              <p className="stat_name">{game.H_Pred_ML}</p>
            </div>
          </div>

          <div
            className={
              shownLineups[game.index]
                ? "lineup-card show"
                : "lineup-card collapse"
            }
          >
            <h2 className="lineup-heading">Projected Lineups</h2>
            <div className="lineup-section">
              <div className="first-team label">
                <p className="stat_name">{game.A_PG}</p>
                <p className="stat_name">{game.A_SG}</p>
                <p className="stat_name">{game.A_SF}</p>
                <p className="stat_name">{game.A_PF}</p>
                <p className="stat_name">{game.A_C}</p>
              </div>
              <div className="positions">
                <p className="stat_name">PG</p>
                <p className="stat_name">SG</p>
                <p className="stat_name">SF</p>
                <p className="stat_name">PF</p>
                <p className="stat_name">C</p>
              </div>
              <div className="second-team label">
                <p className="stat_name">{game.H_PG}</p>
                <p className="stat_name">{game.H_SG}</p>
                <p className="stat_name">{game.H_SF}</p>
                <p className="stat_name">{game.H_PF}</p>
                <p className="stat_name">{game.H_C}</p>
              </div>
            </div>
            <h2 className="lineup-heading lineup-card">Opening Lines</h2>
            <div className="lineup-section">
              <div className="odds_label">
                <p className="stat_name">{game.A_ML_Odds}</p>
                <p className="stat_name">{game.A_S_Odds}</p>
                <p className="stat_name">{game.A_Spread}</p>
                <p className="stat_name">{game.A_Implied + "%"}</p>
                <p className="stat_name">{game.Hold}</p>
              </div>
              <div className="odds_stats">
                <p className="odd_name">Moneyline</p>
                <p className="odd_name">Spread Line</p>
                <p className="odd_name">Spread</p>
                <p className="odd_name">Implied Odds</p>
                <p className="odd_name">Hold</p>
              </div>
              <div className="odds_label">
                <p className="stat_name">{game.H_ML_Odds}</p>
                <p className="stat_name">{game.H_S_Odds}</p>
                <p className="stat_name">{game.H_Spread}</p>
                <p className="stat_name">{game.H_Implied + "%"}</p>
                <p className="stat_name">{game.Hold}</p>
              </div>
            </div>
          </div>
          {/* <div
            className={
              shownLineups[game.index]
                ? "lineup-card show"
                : "lineup-card collapse"
            }
          >
            <h2 className="lineup-heading">Player Status</h2>
            <div className="bottom-section">
              <div className="label">
                <p className="">Danilo Gallinari</p>
                <p className="">Al Horford</p>
                <p className="">Jaylen Brown</p>
              </div>
              <div className="path">
                <p className="">OUT</p>
                <p className="">OUT</p>
                <p className="">TBD</p>
              </div>
              <div className="path">
                <p className="">OUT</p>
                <p className="">OUT</p>
                <p className="">TBD</p>
                <p className="">TBD</p>
              </div>
              <div className="label">
                <p className="">Chris Paul</p>
                <p className="">Jae Crowder</p>
                <p className="">Mikal Bridges</p>
                <p className="">Devin Booker</p>
              </div>
            </div>
          </div> */}
          <span className="collapse-btn">
            {shownLineups[game.index] ? (
              <BsArrowUpCircle onClick={() => toggleLineup(game.index)} />
            ) : (
              <BsArrowDownCircle onClick={() => toggleLineup(game.index)} />
            )}
          </span>
        </div>
      ))}
    </>
  );
};

export default GameCard;
