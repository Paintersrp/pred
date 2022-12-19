import { useState, useEffect } from "react";
import { GameCard, TmwCard } from "../..";
import axios from "axios";
import "./CardDisplay.css";

const GameCards = () => {
  const today = new Date();
  const tomorrow = new Date(today);
  tomorrow.setDate(tomorrow.getDate() + 1);

  const [games, setGames] = useState("");
  const [tmwGames, setTmwGames] = useState("");

  useEffect(() => {
    async function awaitFetch() {
      const response = fetch("./data/cards.json").then((result) =>
        result.json()
      );
      const data = await response;
      setGames(data);
    }

    if (!games) {
      awaitFetch();
    }
  }, []);

  useEffect(() => {
    async function awaitTmwFetch() {
      const responseTmw = fetch("./data/tmw_cards.json").then((result) =>
        result.json()
      );
      const dataTmw = await responseTmw;
      setTmwGames(dataTmw);
    }

    if (!tmwGames) {
      awaitTmwFetch();
    }
  }, []);

  return (
    <section className="layout flex flex-col">
      <h2 className="text-white text-center mt-8 text-xl font-[500] border-b justify-center container date-head">
        {today.toLocaleDateString("en-us", {
          weekday: "long",
          year: "numeric",
          month: "short",
          day: "numeric",
        })}
      </h2>
      <div className="container">
        {games ? <GameCard games={games} /> : null}
      </div>
      <div className="border-b justify-center container date-head flex-col flex gap-3">
        <h4 className="text-white text-center text-xl font-[500] gap-0">
          {tomorrow.toLocaleDateString("en-us", {
            weekday: "long",
            year: "numeric",
            month: "short",
            day: "numeric",
          })}
        </h4>
      </div>
      <div className="container">
        {tmwGames ? <TmwCard games={tmwGames} /> : null}
      </div>
      <div className="container p-0 mb-2">
        <p className="text-white text-[12px] mb-0 flex disclaimer-align">
          *Tomorrow's Predictions may adjust slightly following the completion
          of Today's Games
        </p>
      </div>
    </section>
  );
};

export default GameCards;
