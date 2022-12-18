import React, { useState } from "react";
import { BsArrowUpCircle, BsArrowDownCircle } from "react-icons/bs";
import "./GameCard.css";

const TmwCard = (props) => {
  const currentGamesTmw = props.games;

  if (!currentGamesTmw) {
    console.log("Waiting");
  }

  if (currentGamesTmw) {
    console.log(currentGamesTmw);
  }

  return (
    <>
      {currentGamesTmw.map((game) => (
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
            <div className="label mb-4">
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
        </div>
      ))}
    </>
  );
};

export default TmwCard;
